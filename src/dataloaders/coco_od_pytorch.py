import torch
import torchvision.transforms as T
from torchvision import datasets, ops
from torch.nn.utils.rnn import pad_sequence


def pad_bboxes_with_mask(bboxes_list, max_boxes, pad_value=0.0):
    """
    Pads bounding boxes to a fixed size and returns a mask.

    Args:
        bboxes_list (list of torch.Tensor): List of (num_boxes, 4) tensors.
        max_boxes (int): Fixed number of boxes to pad/truncate to.
        pad_value (float, optional): Padding value. Defaults to 0.0.

    Returns:
        tuple: (padded_boxes, mask)
            - padded_boxes (torch.Tensor): (batch_size, max_boxes, 4)
            - mask (torch.Tensor): (batch_size, max_boxes), 1 for real boxes, 0 for padded ones.
    """
    device = bboxes_list[0].device if bboxes_list else torch.device("cpu")

    # Pad sequences to the longest sequence in the batch
    padded_boxes = pad_sequence(bboxes_list, batch_first=True, padding_value=pad_value)

    # Ensure it has max_boxes length
    if padded_boxes.shape[1] > max_boxes:
        padded_boxes = padded_boxes[:, :max_boxes, :]
    elif padded_boxes.shape[1] < max_boxes:
        extra_pad = torch.full(
            (padded_boxes.shape[0], max_boxes - padded_boxes.shape[1], 4),
            pad_value,
            device=device,
        )
        padded_boxes = torch.cat([padded_boxes, extra_pad], dim=1)

    # Create mask (1 for real boxes, 0 for padded ones)
    mask = (padded_boxes[:, :, 0] != pad_value).float()

    return padded_boxes, mask


def pad_classes(class_list, max_classes, pad_value=0):
    """
    Pads class labels to a fixed size.

    Args:
        class_list (list of torch.Tensor): List of (num_classes,) tensors.
        max_classes (int): Fixed number of classes to pad/truncate to.
        pad_value (int, optional): Padding value for classes.

    Returns:
        tuple: (padded_classes, mask)
            - padded_classes (torch.Tensor): (batch_size, max_classes)
    """
    device = class_list[0].device if class_list else torch.device("cpu")

    # Pad sequences to the longest in batch
    padded_classes = pad_sequence(class_list, batch_first=True, padding_value=pad_value)

    # Ensure it has max_classes length
    if padded_classes.shape[1] > max_classes:
        padded_classes = padded_classes[:, :max_classes]
    elif padded_classes.shape[1] < max_classes:
        extra_pad = torch.full(
            (padded_classes.shape[0], max_classes - padded_classes.shape[1]),
            pad_value,
            device=device,
        )
        padded_classes = torch.cat([padded_classes, extra_pad], dim=1)

    return padded_classes


def preproc_coco(
    annotation: dict, im_w: int, im_h: int, max_boxes=100, empty_class_id=0
):
    """Pre-processing function to unpack a COCO annotation"

    Args:
        annotation (dict): COCO annotation
        im_w (int): Width of the image
        im_h (int): Height of the image

    Returns:
        tuple: (classes, boxes, mask)
            - classes (torch.Tensor): (max_boxes,)
            - boxes (torch.Tensor): (max_boxes, 4)
            - mask (torch.Tensor): (max_boxes,)
    """
    anno = [obj for obj in annotation if "iscrowd" not in obj or obj["iscrowd"] == 0]

    # Get the boxes
    boxes = [obj["bbox"] for obj in anno]
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    # Get the labels for each box
    classes = [obj["category_id"] for obj in anno]
    classes = torch.tensor(classes, dtype=torch.int64)

    # Convert from xywh to xyxy and normalize
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=im_w)
    boxes[:, 1::2].clamp_(min=0, max=im_h)

    # Filter out invalid boxes where x1 > x2 or y1 > y2
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]

    # Normalize the boxes (x1, y1, x2, y2)
    boxes[:, 0::2] /= im_w
    boxes[:, 1::2] /= im_h
    boxes.clamp_(min=0, max=1)

    # Pad the boxes to a maximum size... (example padding [..., [-1, -1, -1, -1]])
    # Mask is a tensor of 1s and 0s (1 for real boxes, 0 for padded ones)
    boxes, padding_mask = pad_bboxes_with_mask(
        [boxes], max_boxes=max_boxes, pad_value=-1
    )

    classes = pad_classes([classes], max_classes=max_boxes, pad_value=empty_class_id)

    # Convert from xyxy to cxcywh
    boxes = ops.box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")

    return classes.squeeze(), boxes.squeeze(), padding_mask.squeeze()


class TorchCOCOLoader(datasets.CocoDetection):
    """ "
    Loader for COCO dataset using Pytorch's CocoDetection

    NOTE: The ground truths are padded to a fixed shape according to "max_boxes" to
          ensure that all samples have the same size (necessary to not use tuple and
          use torch tensors instead).

    Docs:
    - https://pytorch.org/vision/main/generated/torchvision.datasets.CocoDetection.html

    Returns:
        tuple: Tuple of (image, (classes, boxes, padding_mask)) for each objects
    """

    def __init__(
        self,
        root,
        annFile,
        max_boxes=100,
        empty_class_id=0,
        image_size=480,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        super().__init__(
            root,
            annFile,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        # Custom parameters
        self.max_boxes = max_boxes
        self.empty_class_id = empty_class_id
        self.image_size = image_size

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
        image_id = torch.as_tensor([idx], dtype=torch.int64)

        input_ = self.T(img)
        classes, boxes, padding_mask = self.T_target(
            target, w, h, max_boxes=self.max_boxes, empty_class_id=self.empty_class_id
        )

        return input_, (classes, boxes, padding_mask, image_id)


def collate_fn(inputs):
    """
    Collate function for the PyTorch DataLoader.

    Takes a list of items, where each item is a tuple of:
        - input_ (torch.Tensor): The input image tensor.
        - target (tuple): A tuple of (classes, boxes, masks) where:
            - classes (torch.Tensor): The class labels for each object.
            - boxes (torch.Tensor): The bounding boxes for each object.
            - masks (torch.Tensor): The masks for each object.

    Returns:
        tuple: A tuple of (input_, target) where:
            - input_ (torch.Tensor): The batched input image tensor.
            - target (tuple): A tuple of (classes, boxes, masks) where:
                - classes (torch.Tensor): The batched class labels for each object.
                - boxes (torch.Tensor): The batched bounding boxes for each object.
                - masks (torch.Tensor): The batched masks for each object.
    """
    input_ = torch.stack([i[0] for i in inputs])
    classes = torch.stack([i[1][0] for i in inputs])
    boxes = torch.stack([i[1][1] for i in inputs])
    masks = torch.stack([i[1][2].to(dtype=torch.long) for i in inputs])
    image_ids = torch.stack([i[1][3] for i in inputs])
    return input_, (classes, boxes, masks, image_ids)
