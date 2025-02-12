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


def compute_sample_loss(o_bbox, t_bbox, o_cl, t_cl, n_queries=100, empty_class_id=91):

    # If  the example has none box, we just feed it with empty classes.
    if len(t_cl) > 0:
        t_bbox = t_bbox.cuda()
        t_cl = t_cl.cuda()

        o_probs = o_cl.softmax(dim=-1)

        # Negative sign here because we want the maximum magnitude
        C_classes = -o_probs[..., t_cl]

        # Positive sign here because we want to shrink the l1-norm
        C_boxes = torch.cdist(o_bbox, t_bbox, p=1)

        # Negative sign here because we want the maximum magnitude
        C_giou = -ops.generalized_box_iou(
            ops.box_convert(o_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
            ops.box_convert(t_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
        )

        C_total = 1 * C_classes + 5 * C_boxes + 2 * C_giou

        # Convert the tensor to numpy array
        C_total = C_total.cpu().detach().numpy()

        # Find the optimum pairs that produces the minimum summation.
        # the method returns the pair indices
        o_ixs, t_ixs = linear_sum_assignment(C_total)

        # Transform indices to tensors
        o_ixs = torch.IntTensor(o_ixs)
        t_ixs = torch.IntTensor(t_ixs)

        # Reorder o_ixs to naturally align with target_cl length, such
        # the pairs are {(o_ixs[0], t[0]), {o_ixs[1], t[1]}, ...}
        o_ixs = o_ixs[t_ixs.argsort()]

        # Average over the number of boxes, not the number of coordinates...
        num_boxes = len(t_bbox)
        loss_bbox = F.l1_loss(o_bbox[o_ixs], t_bbox, reduce="sum") / num_boxes

        # Get the GIoU matrix
        target_gIoU = ops.generalized_box_iou(
            ops.box_convert(o_bbox[o_ixs], in_fmt="cxcywh", out_fmt="xyxy"),
            ops.box_convert(t_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
        )

        # Get only the matrix diagonal from the GIoU matrix (num_queries, num_gt_boxes) that contains
        # the bipartite pairs and transform gIoU into a loss (1- GIoU).
        loss_giou = 1 - torch.diag(target_gIoU).mean()

        # Calculate the class cross-entropy (pad with 91 (empty for COCO) for non-existent labels)
        queries_classes_label = torch.full(o_probs.shape[:1], empty_class_id).cuda()
        queries_classes_label[o_ixs] = t_cl
        loss_class = F.cross_entropy(o_cl, queries_classes_label)

    else:
        queries_classes_label = torch.full((n_queries,), empty_class_id).cuda()
        loss_class = F.cross_entropy(o_cl, queries_classes_label)
        loss_bbox = loss_giou = torch.tensor(0)

    return loss_class, loss_bbox, loss_giou
