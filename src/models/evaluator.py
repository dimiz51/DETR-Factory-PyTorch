import torch
import numpy as np
import pycocotools.coco as coco
from torchvision import datasets
import pycocotools.cocoeval as cocoeval
from torch.utils.data import DataLoader
from utils.inference import class_based_nms
from torchvision import ops


class DETREvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        coco_dataset: datasets.CocoDetection,
        device: torch.device,
        empty_class_id: int,
        collate_fn: callable,
        nms_iou_threshold: float = 0.5,
    ):
        """
        Evaluator for DETR using COCO evaluation metrics.

        Args:
            model (torch.nn.Module): The trained DETR model.
            coco_dataset (torch.utils.data.Dataset): The COCO dataset used for evaluation.
            device (torch.device): The device to run the model on.
            empty_class_id (int): The class ID for the empty class (background).
            collate_fn (callable): The collate function for the DataLoader.
            nms_iou_threshold (float, optional): The IOU threshold for NMS. Defaults to 0.5.
        """
        self.model = model.to(device)
        self.device = device
        self.coco_gt = coco_dataset.coco  # COCO ground truth annotations
        self.empty_class_id = empty_class_id
        self.nms_iou_threshold = nms_iou_threshold

        # Create DataLoader with batch_size=1 (required for COCO eval) and no shuffling
        self.dataloader = DataLoader(
            coco_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
        )

    def evaluate(self):
        """
        Runs evaluation on the dataset and computes COCO metrics.
        """
        self.model.eval()
        results = []

        # Evaluation dataset information
        print(f"Number of images in evaluation COCO dataset: {len(self.coco_gt.imgs)}")
        print(
            f"Number of objects in the evaluation COCO dataset: {len(self.coco_gt.anns)}"
        )

        print(f"Evaluating DETR model on device: {self.device}")

        with torch.no_grad():
            for ix, (input_, (_, _, _, image_ids)) in enumerate(self.dataloader):
                # # Load the image size from the coco annotation
                # img_size = (self.coco_gt.imgs[img_idx[0].item()]["width"], self.coco_gt.imgs[img_idx[0].item()]["height"])
                # # If widht/height are not the same, throw an error...
                # # Evaluation for not square images is not supported yet...
                # if img_size[0] != img_size[1]:
                #     raise ValueError("Evaluation for not square images is not supported yet...")

                # Move inputs to device
                input_ = input_.to(self.device)
                outputs = self.model(input_)
                out_cl, out_bbox = outputs["layer_5"].values()

                # Squeeze and apply activations
                out_bbox = out_bbox.sigmoid().cpu()
                out_cl = out_cl.cpu()

                batch_size = input_.shape[0]

                # Process each image in the batch...
                for img_idx in range(batch_size):
                    o_probs = out_cl[img_idx].softmax(dim=-1)
                    o_bbox = out_bbox[img_idx]
                    img_id = image_ids[img_idx].item()

                    # Get ground truth image size for rescaling...
                    gt_image_size = (
                        self.coco_gt.imgs[img_id]["width"],
                        self.coco_gt.imgs[img_id]["height"],
                    )
                    if gt_image_size[0] != gt_image_size[1]:
                        raise ValueError(
                            "Evaluation for non square images is not supported yet..."
                        )

                    # Convert bounding boxes to COCO format()
                    o_bbox = ops.box_convert(
                        o_bbox * gt_image_size[0], in_fmt="cxcywh", out_fmt="xywh"
                    )

                    # Filter out "no object" predictions
                    o_keep = o_probs.argmax(dim=-1) != self.empty_class_id

                    # If no object is predicted, skip this image
                    if o_keep.sum() == 0:
                        continue

                    # Filter out "empty box" predictions
                    keep_boxes = o_bbox[o_keep]
                    keep_probs = o_probs[o_keep]

                    # Apply class-based NMS
                    nms_boxes, nms_probs, nms_classes = class_based_nms(
                        keep_boxes, keep_probs, iou_threshold=self.nms_iou_threshold
                    )

                    # Convert to COCO format
                    for j in range(len(nms_classes)):
                        results.append(
                            {
                                "image_id": img_id,
                                "category_id": nms_classes[j].item(),
                                "bbox": nms_boxes[j].tolist(),
                                "score": nms_probs[j].item(),
                            }
                        )

        if len(results) == 0:
            raise ValueError(
                "No objects were found, something could be wrong with the model provided!"
            )

        # Create COCO results object
        coco_dt = self.coco_gt.loadRes(results)

        # Initialize COCO evaluator
        coco_eval = cocoeval.COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats
