import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.nn.utils.rnn import pad_sequence
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import torch
from torchvision.ops import box_convert


def augment_transforms_A(crop_width=480, crop_height=480):
    """
    Applies a series of data augmentation transformations to images and their corresponding bounding boxes.

    Args:
        crop_width (int, optional): Width of the cropped image. Default is 480.
        crop_height (int, optional): Height of the cropped image. Default is 480.

    Returns:
        A.Compose: An Albumentations composition object that applies the following transformations:
            - Randomly applies either HueSaturationValue or RandomBrightnessContrast with a probability of 0.9.
            - Converts the image to grayscale with a probability of 0.01.
            - Horizontally flips the image with a probability of 0.5.
            - Vertically flips the image with a probability of 0.5.
            - Applies a cutout operation with 8 holes of maximum size 64x64 and a fill value of 0, with a probability of 0.5.
            - Converts the image to a tensor suitable for PyTorch models.

        The bounding box parameters are set to use the YOLO format, with no minimum area or visibility requirements,
        and the label fields are specified as "labels".
    """
    return A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.6
                    ),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
                ],
                p=0.7,
            ),
            A.RandomSizedBBoxSafeCrop(width=crop_width, height=crop_height, erosion_rate=0.2, p = 0.15),
            A.HorizontalFlip(p=0.35),
            A.VerticalFlip(p=0.35),
            ToTensorV2(p=1),
        ],
        bbox_params=A.BboxParams(
            format="albumentations",
            min_area=0,
            min_visibility=0,
            label_fields=["labels"],
        ),
    )


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


def preproc_target(annotation, max_boxes=100, empty_class_id=0):
    """Pre-process COCO annotations: extract boxes and classes, pad and change format.

    Args:
        annotation (list): List of COCO annotations.
        max_boxes (int): Max number of boxes (default: 100).
        empty_class_id (int): Padding value for empty class (default: 0).

    Returns:
        tuple:
            - classes (torch.Tensor): (max_boxes,)
            - boxes (torch.Tensor): (max_boxes, 4) in cxcywh format.
            - mask (torch.Tensor): (max_boxes,) indicating real vs. padded boxes.
    """
    # Extract boxes & labels
    boxes = torch.tensor(
        [obj["bbox"] for obj in annotation], dtype=torch.float32
    ).reshape(-1, 4)
    classes = torch.tensor(
        [obj["category_id"] for obj in annotation], dtype=torch.int64
    )

    # Handle empty case early
    if len(boxes) == 0:
        return (
            torch.full(
                (max_boxes,), empty_class_id, dtype=torch.int64
            ),  # Padded classes
            torch.zeros((max_boxes, 4), dtype=torch.float32),  # Zero boxes
            torch.zeros((max_boxes,), dtype=torch.uint8),  # Mask: all padded
        )

    # Pad boxes & classes
    boxes, padding_mask = pad_bboxes_with_mask(
        [boxes], max_boxes=max_boxes, pad_value=0
    )
    classes = pad_classes([classes], max_classes=max_boxes, pad_value=empty_class_id)

    # Convert xyxy → cxcywh
    boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")

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
        albumentation_transforms=None,
        augment=False,
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
        self.augment = augment
        self.albumentations_transforms = augment_transforms_A(image_size, image_size) if albumentation_transforms is None else albumentation_transforms

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
        self.T_target = preproc_target

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        w, h = img.size
        image_id = torch.tensor([idx], dtype=torch.int64)

        # Convert to NumPy for Albumentations
        img_np = np.array(img)

        # Remove crowd annotations...
        target = [obj for obj in target if obj.get("iscrowd", 0) == 0]

        # Extract raw boxes & labels before processing
        raw_boxes = torch.tensor(
            [obj["bbox"] for obj in target], dtype=torch.float32
        ).reshape(-1, 4)
        raw_classes = torch.tensor(
            [obj["category_id"] for obj in target], dtype=torch.int64
        )

        # Convert xywh → xyxy and normalize
        if raw_boxes.numel() > 0:
            raw_boxes[:, 2:] += raw_boxes[:, :2]
            raw_boxes[:, 0::2] /= w
            raw_boxes[:, 1::2] /= h
            raw_boxes.clamp_(0, 1)
        else:
            raw_boxes = torch.empty((0, 4), dtype=torch.float32)
            raw_classes = torch.empty((0,), dtype=torch.int64)

        # Apply Albumentations (before padding)
        if self.augment and self.albumentations_transforms:
            transformed = self.albumentations_transforms(
                image=img_np, bboxes=raw_boxes.tolist(), labels=raw_classes.tolist()
            )
            img_np = transformed["image"]
            raw_boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            raw_classes = torch.tensor(transformed["labels"], dtype=torch.int64)

        # Apply `preproc_coco` for padding & cxcywh conversion
        classes, boxes, padding_mask = self.T_target(
            [
                {"bbox": b.tolist(), "category_id": c.item()}
                for b, c in zip(raw_boxes, raw_classes)
            ],
            max_boxes=self.max_boxes,
            empty_class_id=self.empty_class_id,
        )

        # Transform Image (Resize, Normalize)
        input_ = self.T(T.ToPILImage()(img_np))

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
