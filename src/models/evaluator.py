import torch
from torchvision import datasets
import pycocotools.cocoeval as cocoeval
from torch.utils.data import DataLoader
from utils.inference import run_inference
import numpy as np


class DETREvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        coco_dataset: datasets.CocoDetection,
        device: torch.device,
        empty_class_id: int,
        collate_fn: callable,
        nms_iou_threshold: float = 0.5,
        batch_size: int = 2,
        image_size: int = 480,
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
        self.image_size = image_size

        # Create DataLoader and no shuffling
        self.dataloader = DataLoader(
            coco_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
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
                # Extract inference results
                batch_results = run_inference(
                    self.model,
                    self.device,
                    input_,
                    nms_threshold=self.nms_iou_threshold,
                    image_size=self.image_size,
                    empty_class_id=self.empty_class_id,
                    out_format="xywh",  # COCO format for boxes
                    scale_boxes=False,  # We don't want to scale to inference image size as those might differ from the COCO ground truths
                )

                # Process each image in the batch...
                for img_idx, (nms_boxes, nms_probs, nms_classes) in enumerate(
                    batch_results
                ):
                    img_id = image_ids[img_idx].item()

                    # Skip images where no objects are detected
                    if len(nms_boxes) == 0:
                        continue

                    # Get the scaling factors
                    scale_factors = np.array(
                        [
                            self.coco_gt.imgs[img_id]["width"],
                            self.coco_gt.imgs[img_id]["height"],
                            self.coco_gt.imgs[img_id]["width"],
                            self.coco_gt.imgs[img_id]["height"],
                        ],
                        dtype=np.float32,
                    )

                    # Scale the boxes to image size...
                    nms_boxes = nms_boxes * scale_factors

                    # Convert detections to COCO format
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
