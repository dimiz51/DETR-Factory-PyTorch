import torch
import torchvision.transforms as T
from torchvision import datasets, ops


def preproc_coco(annotation: dict, im_w: int, im_h: int):
    """Pre-processing function to unpack a COCO annotation"

    Args:
        annotation (dict): COCO annotation
        im_w (int): Width of the image
        im_h (int): Height of the image

    Returns:
        tuple: Tuple of (classes, boxes)
    """
    anno = [obj for obj in annotation if "iscrowd" not in obj or obj["iscrowd"] == 0]

    boxes = [obj["bbox"] for obj in anno]
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    # Convert from xywh to xyxy and normalize
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=im_w)
    boxes[:, 1::2].clamp_(min=0, max=im_h)

    # Filter out invalid boxes where x1 > x2 or y1 > y2
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]

    # Get the labels for each box
    classes = [obj["category_id"] for obj in anno]
    classes = torch.tensor(classes, dtype=torch.int64)
    classes = classes[keep]

    # Normalize the boxes (x1, y1, x2, y2)
    boxes[:, 0::2] /= im_w
    boxes[:, 1::2] /= im_h
    boxes.clamp_(min=0, max=1)

    # Convert from xyxy to cxcywh
    boxes = ops.box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")

    return classes, boxes


class TorchCOCOLoader(datasets.CocoDetection):
    """ "
    Loader for COCO dataset using Pytorch's CocoDetection

    Docs:
    - https://pytorch.org/vision/main/generated/torchvision.datasets.CocoDetection.html

    Returns:
        tuple: Tuple of (image, (classes, boxes)) for each item
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = 480

        # Transformations pipeline
        self.T = T.Compose(
            [
                T.ToTensor(),
                # We need this normalization as our CNN backbone
                # is trained on ImageNet:
                # - https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.Resize((self.image_size, self.image_size), antialias=True),
            ]
        )

        # Transformation function
        self.T_target = preproc_coco

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        w, h = img.size

        input_ = self.T(img)
        classes, boxes = self.T_target(target, w, h)

        return input_, (classes, boxes)


def collate_fn(inputs):
    input_ = torch.stack([i[0] for i in inputs])
    classes = tuple([i[1][0] for i in inputs])
    boxes = tuple([i[1][1] for i in inputs])
    return input_, (classes, boxes)
