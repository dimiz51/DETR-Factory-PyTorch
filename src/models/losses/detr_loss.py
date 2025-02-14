## The DETR loss

# The loss in the DETR model is a weighted sum of the Generalized IoU Loss(GIoU), the L1 loss from the box coordinates and the classification loss for each box.
# Additionally, we have the "set prediction problem" where we have n possible object matches (see object queries) at maximum.
# In order to map a query to a box from the ground truth so we can minimize the loss we apply the Hungarian algorithm (or scipy's `linear_sum_assignment`).
# This algorithm attempts to minimize a cost matrix (weighted sum of losses) and outputs the optimal pairs to achieve that.
# This way we can map the best candidates between the object queries and the ground truth boxes.
# The default weighting factors are 1 for the class loss, 5 for the GIoU loss and 2 for the box L1-loss.

from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
import torch
import torchvision.ops as ops


def compute_sample_loss(
    o_bbox, t_bbox, o_cl, t_cl, t_mask, n_queries=100, empty_class_id=91, device="cuda"
):
    """Compute a the DETR loss for a single sample

    Args:
        o_bbox (torch.Tensor): The predicted bounding boxes (Shape: torch.Size([100, 4]))
        t_bbox (torch.Tensor): The ground truth bounding boxes (Shape: torch.Size([100, 4]))
        o_cl (torch.Tensor): The predicted class labels (Shape: torch.Size([100, num_classes]))
        t_cl (torch.Tensor): The ground truth class labels (Shape: torch.Size([100]))
        t_mask (torch.Tensor): The mask(boolean) for the ground truth bounding boxes (Shape: torch.Size([100]))
        n_queries (int, optional): The number of object queries. Defaults to 100.
        empty_class_id (int, optional): The class ID representing 'no object'. Defaults to 91.

    Returns:
        tuple: Tuple of (loss_class, loss_bbox, loss_giou)

    """

    # Filter out the padded classes/boxes using the boolean mask
    valid_gt_boxes = t_bbox[t_mask]
    valid_gt_classes = t_cl[t_mask]

    # If we don't have any ground truth actual objects left we can return early....
    if valid_gt_boxes.numel() == 0:
        queries_classes_label = torch.full((n_queries,), empty_class_id, device=device)
        loss_class = F.cross_entropy(o_cl, queries_classes_label)
        loss_bbox = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)
        return loss_class, loss_bbox, loss_giou

    # Else calculate the losses....
    o_probs = o_cl.softmax(dim=-1)

    # Negative sign here because we want the maximum magnitude
    C_classes = -o_probs[..., valid_gt_classes]

    # Positive sign here because we want to shrink the l1-norm
    C_boxes = torch.cdist(o_bbox, valid_gt_boxes, p=1)

    # Negative sign here because we want the maximum magnitude
    C_giou = -ops.generalized_box_iou(
        ops.box_convert(o_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
        ops.box_convert(valid_gt_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
    )

    C_total = 1 * C_classes + 5 * C_boxes + 2 * C_giou

    # Convert the tensor to numpy array
    C_total = C_total.cpu().detach().numpy()

    # Find the optimum pairs that produces the minimum summation.
    # the method returns the pair indices
    o_ixs, t_ixs = linear_sum_assignment(C_total)

    # Transform indices to tensors
    o_ixs = torch.as_tensor(o_ixs, dtype=torch.long, device=device)
    t_ixs = torch.as_tensor(t_ixs, dtype=torch.long, device=device)

    # Reorder `o_ixs` to align with `t_cl` length

    # Reorder o_ixs to naturally align with target_cl length, such
    # the pairs are {(o_ixs[0], t[0]), {o_ixs[1], t[1]}, ...}
    o_ixs = o_ixs[t_ixs.argsort()]

    # Average over the number of boxes
    num_boxes = len(valid_gt_boxes)

    # Compute the L1 loss for the boxes..

    loss_bbox = F.l1_loss(o_bbox[o_ixs], valid_gt_boxes, reduction="sum") / num_boxes

    # Get the GIoU matrix
    target_gIoU = ops.generalized_box_iou(
        ops.box_convert(o_bbox[o_ixs], in_fmt="cxcywh", out_fmt="xyxy"),
        ops.box_convert(valid_gt_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
    )

    # Compute GIoU loss

    # Get only the matrix diagonal from the GIoU matrix (num_queries, num_gt_boxes) that contains
    # the bipartite pairs and transform gIoU into a loss (1- GIoU).
    loss_giou = 1 - torch.diag(target_gIoU).mean()

    # Calculate the class cross-entropy (pad with 91 (empty for COCO) for non-existent labels)
    queries_classes_label = torch.full(o_probs.shape[:1], empty_class_id, device=device)
    queries_classes_label[o_ixs] = valid_gt_classes

    # Compute classification loss
    loss_class = F.cross_entropy(o_cl, queries_classes_label)

    return loss_class, loss_bbox, loss_giou
