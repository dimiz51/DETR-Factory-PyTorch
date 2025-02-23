import torch
from torchvision.ops import nms
import torchvision.ops as ops
import numpy as np


def class_based_nms(boxes, probs, iou_threshold=0.5):
    """
    Performs non-maximum suppression (NMS) on bounding boxes to filter out overlapping
    boxes for each class. This is usually not needed for DETR as it usually does not produce
    overlapping boxes (if trained long enough).

    Args:
        boxes (torch.Tensor): Bounding boxes in the format (xmin, ymin, xmax, ymax). Shape: [num_boxes, 4]
        probs (torch.Tensor): Class probabilities for each bounding box. [num_boxes, num_classes]
        iou_threshold (float, optional): IOU threshold for NMS. Defaults to 0.5.

    Returns:
        torch.Tensor: Bounding boxes after NMS.
        torch.Tensor: Predicted class scores after NMS.
        torch.Tensor: Predicted class indices after NMS.
    """

    # Get the class with the highest probability for each box
    scores, class_ids = torch.max(probs, dim=1)

    # Apply NMS
    keep_ids = nms(boxes, scores, iou_threshold)

    # Get the boxes and class scores after NMS
    boxes = boxes[keep_ids]
    scores = scores[keep_ids]
    class_ids = class_ids[keep_ids]

    return boxes, scores, class_ids


def run_inference(
    model,
    device,
    inputs,
    nms_threshold=0.3,
    image_size=480,
    empty_class_id=0,
    out_format="xyxy",
    scale_boxes=True,
):
    """
    Utility function that wraps the inference and post-processing and returns the results for the
    batch of inputs. The inference will be run using the passed model and device while post-processing
    will be done on the CPU.

    Args:
        model (torch.nn.Module): The trained model for inference.
        device (torch.device): The device to run inference on.
        inputs (torch.Tensor): Batch of input images.
        nms_threshold (float, optional): NMS threshold for removing overlapping boxes. Default is 0.3.
        image_size (int, optional): Image size for transformations. Default is 480.
        empty_class_id (int, optional): The class ID representing 'no object'. Default is 0.
        out_format (str, optional): Output format for bounding boxes. Default is "xyxy".
        scale_boxes (bool, optional): Whether to scale the bounding boxes. Default is True.
    Returns:
        List of tuples: Each tuple contains (nms_boxes, nms_probs, nms_classes) for a batch item.
    """
    if model and device:
        model.eval()
        model.to(device)
        inputs = inputs.to(device)
    else:
        raise ValueError("No model or device provided for inference!")

    with torch.no_grad():
        out_cl, out_bbox = model(inputs)

    # Get the outputs from the last decoder layer..
    out_cl = out_cl[:, -1, :]
    out_bbox = out_bbox[:, -1, :]
    out_bbox = out_bbox.sigmoid().cpu()
    out_cl_probs = out_cl.cpu()

    scale_factors = torch.tensor([image_size, image_size, image_size, image_size])
    results = []

    for i in range(inputs.shape[0]):
        o_bbox = out_bbox[i]
        o_cl = out_cl_probs[i].softmax(dim=-1)
        o_bbox = ops.box_convert(o_bbox, in_fmt="cxcywh", out_fmt=out_format)

        # Scale boxes if needed...
        if scale_boxes:
            o_bbox = o_bbox * scale_factors

        # Filter out boxes with no object...
        o_keep = o_cl.argmax(dim=-1) != empty_class_id
        if o_keep.sum() == 0:
            results.append((np.array([]), np.array([]), np.array([])))
            continue
        keep_probs = o_cl[o_keep]
        keep_boxes = o_bbox[o_keep]

        # Apply NMS
        nms_boxes, nms_probs, nms_classes = class_based_nms(
            keep_boxes, keep_probs, nms_threshold
        )
        results.append((nms_boxes, nms_probs, nms_classes))

    return results
