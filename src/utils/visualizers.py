import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import ops
import torch
import os

# Default colors for visualization of boxes
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]
COLORS *= 100  # Repeat colors to cover all classes


class DETRBoxVisualizer:
    def __init__(self, class_labels, empty_class_id, normalization_params=(None, None)):
        """
        The DETR box visualizer is responsible for visualizing the inputs/outputs of the DETR model.

        You can use the public API of the class to:
        - Visualize a single image with "visualize_image()"
        - Visualize inferene results from a validation batch with "visualize_validation_inference()"
        - Visualize a single image with inference results with "visualize_inference()"

        Args:
            class_labels (list): List of class labels.
            normalization_params (tuple): Mean and standard deviation used for normalization.
            empty_class_id (int): The class ID representing 'no object'.
        """
        self.class_labels = class_labels
        self.empty_class_id = empty_class_id

        if normalization_params != (None, None):
            self.normalization_params = normalization_params
        else:
            # Assume ImageNet normalization
            self.normalization_params = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the unnormalize transform
        mean, std = self.normalization_params
        self.unnormalize = T.Normalize(
            mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
        )

    def _revert_normalization(self, tensor):
        """
        Reverts the normalization of an image tensor.

        Args:
            tensor (torch.Tensor): Normalized image tensor.

        Returns:
            torch.Tensor: Denormalized image tensor.
        """
        return self.unnormalize(tensor)

    def _visualize_image(self, im, boxes, probs=None, ax=None):
        """
        Visualizes a single image with bounding boxes and predicted probabilities.
        NOTE: The boxes tensors is expected to be in the format (xmin, ymin, xmax, ymax) and
              in pixel space already (not normalized).

        Args:
            im (np.array): Image to visualize.
            boxes (np.array): Bounding boxes.
            probs (np.array, optional): Probabilities for each box.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object.
        """
        if ax is None:
            ax = plt.gca()

        # Revert normalization for image
        im = self._revert_normalization(im).permute(1, 2, 0).cpu().clip(0, 1)

        ax.imshow(im)
        ax.axis("off")  # Hide axes

        for i, b in enumerate(boxes.tolist()):
            xmin, ymin, xmax, ymax = b
            patch = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color=COLORS[i],
                linewidth=2,
            )
            ax.add_patch(patch)
            if probs is not None:
                if probs.ndim == 1:
                    cl = probs[i].item()
                    text = f"{self.class_labels[cl]}"
                else:
                    cl = probs[i].argmax().item()
                    text = f"{self.class_labels[cl]}: {probs[i, cl]:0.2f}"
                ax.text(
                    xmin,
                    ymin,
                    text,
                    fontsize=7,
                    bbox=dict(facecolor="yellow", alpha=0.5),
                )

    def visualize_validation_inference(
        self, model, dataset, batch_size=2, collate_fn=None, image_size=480
    ):
        """
        Performs inference on the validation dataset and visualizes predictions.

        Args:
            model (torch.nn.Module): The trained model for inference.
            dataset (torch.utils.data.Dataset): The dataset to perform inference on.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 2.
            collate_fn(fn, optional): Collate function to create a dataloader from the dataset
            image_size(int, optional): The image size of the images in the dataset (Default: 480)
        """
        if model:
            model = model.eval()
            model.to(self.device)
        else:
            raise ValueError("No model provided for inference!")

        if dataset is None:
            raise ValueError("No validation dataset provided for inference!")

        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        inputs, (tgt_cl, tgt_bbox, tgt_mask) = next(iter(data_loader))

        # Move inputs to GPU if available and run inference
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = model(inputs)
        print(f"Running inference on device: {self.device}")

        # Assuming 'layer_5' contains the desired outputs
        out_cl, out_bbox = outputs["layer_5"].values()
        out_bbox = out_bbox.sigmoid().cpu()
        out_cl = out_cl.cpu()

        fig, axs = plt.subplots(
            batch_size, 2, figsize=(15, 7.5 * batch_size), constrained_layout=True
        )
        if batch_size == 1:
            axs = axs[np.newaxis, :]

        for ix in range(batch_size):
            # Get true and predicted boxes for the batch
            o_cl = out_cl[ix]
            t_cl = tgt_cl[ix]
            o_bbox = out_bbox[ix]
            t_bbox = tgt_bbox[ix]
            t_mask = tgt_mask[ix].bool()

            # Filter out empty boxes from the ground truths
            t_cl = t_cl[t_mask]
            t_bbox = t_bbox[t_mask]

            # Apply softmax and rescale boxes
            o_probs = o_cl.softmax(dim=-1)
            o_bbox = ops.box_convert(
                o_bbox * image_size, in_fmt="cxcywh", out_fmt="xyxy"
            )
            t_bbox = ops.box_convert(
                t_bbox * image_size, in_fmt="cxcywh", out_fmt="xyxy"
            )

            # Filter "no object" predictions
            o_keep = o_probs.argmax(-1) != self.empty_class_id

            # Plot image with predictions on the left
            self._visualize_image(
                inputs[ix].cpu(), o_bbox[o_keep], o_probs[o_keep], ax=axs[ix, 0]
            )
            axs[ix, 0].set_title("Predictions")

            # Plot image with ground truth boxes on the right
            self._visualize_image(inputs[ix].cpu(), t_bbox, t_cl, ax=axs[ix, 1])
            axs[ix, 1].set_title("Ground Truth")

        plt.show()


def visualize_losses(hist, hist_detail=None, save_dir=None):
    """
    Plots training loss over epochs and optionally saves the figure.

    Args:
        hist (list): List of total loss values per epoch.
        hist_detail (list, optional): List of tuples (class_loss, bbox_loss, giou_loss) per epoch.
        save_dir (str, optional): Directory to save the plots. If None, it only displays the plots.
    """

    # Create save directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    epochs = np.arange(1, len(hist) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, hist, label="Total Loss", marker="o", linestyle="-")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "DETR_training_loss.png"))

    plt.show()

    # If detailed loss is provided, plot them separately
    if hist_detail:
        class_loss, bbox_loss, giou_loss = zip(*hist_detail)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, class_loss, label="Class Loss", linestyle="--")
        plt.plot(epochs, bbox_loss, label="BBox Loss", linestyle="--")
        plt.plot(epochs, giou_loss, label="GIoU Loss", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Detailed Training Loss Over Epochs")
        plt.legend()
        plt.grid()

        if save_dir:
            plt.savefig(os.path.join(save_dir, "DETR_training_losses.png"))

        plt.show()
