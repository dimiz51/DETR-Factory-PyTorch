import torch
from torchvision.ops import nms


def class_based_nms(boxes, probs, iou_threshold=0.5):
    """
    Performs non-maximum suppression (NMS) on bounding boxes based on class probabilities.

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
