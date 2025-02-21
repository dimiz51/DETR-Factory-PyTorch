import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import ops
import torch
from utils.inference import run_inference
import cv2
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
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
        - Visualize a single image or inference results with "visualize_image()"
        - Visualize batch inference results using a validation dataset with "visualize_validation_inference()"
        Args:
            class_labels (list): List of class labels.
            normalization_params (tuple): Mean and standard deviation used for normalization.
            empty_class_id (int): The class ID representing 'no object'.
        """
        self.class_labels = class_labels
        self.empty_class_id = empty_class_id
        self.class_to_color = {}

        if normalization_params != (None, None) and type(normalization_params) == tuple:
            if len(normalization_params) != 2:
                raise ValueError(
                    "Expected normalization_params to be a tuple of length 2!"
                )

            mean, std = normalization_params
            if len(mean) != 3 or len(std) != 3:
                raise ValueError("Expected mean and std to be tuples of length 3!")
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

    def _visualize_image(
        self, im, boxes, class_ids, scores=None, ax=None, show_scores=True
    ):
        """
        Visualizes a single image with bounding boxes and predicted probabilities.
        NOTE: The boxes tensors is expected to be in the format (xmin, ymin, xmax, ymax) and
              in pixel space already (not normalized).

        Args:
            im (np.array): Image to visualize.
            boxes (np.array): Bounding boxes.
            class_ids (np.array): Class IDs for each box.
            scores (np.array, optional): Probabilities for each box.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object.
            show_scores (bool, optional): Whether to show the predicted probabilities.
        """
        if ax is None:
            ax = plt.gca()

        # Revert normalization for image
        im = self._revert_normalization(im).permute(1, 2, 0).cpu().clip(0, 1)

        ax.imshow(im)
        ax.axis("off")  # Hide axes

        for i, b in enumerate(boxes.tolist()):
            xmin, ymin, xmax, ymax = b

            if scores is not None:
                score = scores[i]
            else:
                score = None

            if class_ids is not None:
                cl = class_ids[i]
            else:
                raise ValueError("No class IDs provided for visualization!")

            # Assign a color to the class if not already assigned
            if cl not in self.class_to_color:
                self.class_to_color[cl] = COLORS[cl % len(COLORS)]

            color = self.class_to_color[cl]

            # Draw bounding box
            patch = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color=color,
                linewidth=2,
            )
            ax.add_patch(patch)

            # Add label text
            text = (
                f"{self.class_labels[cl]}"
                if score is None or not show_scores
                else f"{self.class_labels[cl]}: {score:0.2f}"
            )
            ax.text(
                xmin, ymin, text, fontsize=7, bbox=dict(facecolor="yellow", alpha=0.5)
            )

    def visualize_validation_inference(
        self,
        model,
        dataset,
        batch_size=2,
        collate_fn=None,
        image_size=480,
        nms_threshold=0.3,
        output_layer="layer_5",
    ):
        """
        Performs inference on the validation dataset and visualizes predictions.

        Args:
            model (torch.nn.Module): The trained model for inference.
            dataset (torch.utils.data.Dataset): The dataset to perform inference on.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 2.
            collate_fn(fn, optional): Collate function to create a dataloader from the dataset
            image_size(int, optional): The image size of the images in the dataset (Default: 480)
            nms_threshold(float, optional): The threshold for NMS (Default: 0.5)
            output_layer(str, optional): The output layer to use for inference (Default: "layer_5")
        """
        if dataset is None:
            raise ValueError("No validation dataset provided for inference!")

        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        inputs, (tgt_cl, tgt_bbox, tgt_mask, _) = next(iter(data_loader))

        # Move inputs to GPU if available and run inference
        print(f"Running inference on device: {self.device}")
        inference_results = run_inference(
            model=model,
            device=self.device,
            inputs=inputs,
            nms_threshold=nms_threshold,
            image_size=image_size,
            empty_class_id=self.empty_class_id,
            output_layer=output_layer,
        )

        fig, axs = plt.subplots(
            batch_size, 2, figsize=(15, 7.5 * batch_size), constrained_layout=True
        )
        if batch_size == 1:
            axs = axs[np.newaxis, :]

        for ix in range(batch_size):
            # Get true and predicted boxes for the batch
            t_cl = tgt_cl[ix]
            t_bbox = tgt_bbox[ix]
            t_mask = tgt_mask[ix].bool()

            # Filter out empty ground truth boxes
            t_cl = t_cl[t_mask]
            t_bbox = t_bbox[t_mask]

            # Convert to xyxy format
            t_bbox = ops.box_convert(
                t_bbox * image_size, in_fmt="cxcywh", out_fmt="xyxy"
            )

            # Extract inference results
            nms_boxes, nms_probs, nms_classes = inference_results[ix]

            # Plot predictions
            self._visualize_image(
                inputs[ix].cpu(), nms_boxes, nms_classes, nms_probs, ax=axs[ix, 0]
            )
            axs[ix, 0].set_title("Predictions")

            # Plot ground truth
            self._visualize_image(inputs[ix].cpu(), t_bbox, t_cl, ax=axs[ix, 1])
            axs[ix, 1].set_title("Ground Truth")

        plt.show()

    def visualize_video_inference(
        self,
        model,
        video_path,
        save_dir,
        image_size=480,
        batch_size=5,
        nms_threshold=0.3,
        output_layer="layer_5",
    ):
        """
        Processes a video, runs inference in batches of frames, visualizes results, and saves a new video.

        Args:
            model (torch.nn.Module): The trained model for inference.
            video_path (str): Path to the input video.
            save_dir (str): Directory to save the processed video.
            image_size (int, optional): Image size for transformations. Default is 480.
            batch_size (int, optional): Number of frames per inference batch. Default is 5.
            nms_threshold (float, optional): NMS threshold for removing overlapping boxes. Default is 0.3.
            output_layer (str, optional): The output layer to use for inference. Default is "layer_5".
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")

        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Video FPS: {original_fps}")

        transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToTensor(),
                T.Normalize(
                    mean=self.normalization_params[0], std=self.normalization_params[1]
                ),
            ]
        )

        frames = []
        frame_batches = []
        processed_frames = []

        print(f"Running inference on device: {self.device}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Convert OpenCV frame (BGR) to PIL Image (RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)

            # Apply transformations and add batch dimension
            img_tensor = transform(pil_img).unsqueeze(0)

            # Get the original image size
            video_h, video_w, _ = frame.shape

            # Resize original frame to match image size...
            frames.append(frame)
            frame_batches.append(img_tensor)

            # Process batch when we have enough frames
            if len(frame_batches) == batch_size:
                # Build batch for batch inference...
                batch_input = torch.cat(frame_batches, dim=0)

                # Run inference using the specified device...
                inference_results = run_inference(
                    model=model,
                    device=self.device,
                    inputs=batch_input,
                    nms_threshold=nms_threshold,
                    image_size=image_size,
                    empty_class_id=self.empty_class_id,
                    output_layer=output_layer,
                )

                for i in range(batch_size):
                    nms_boxes, nms_probs, nms_classes = inference_results[i]
                    # If there are no boxes just add the original frame and continue...
                    if nms_boxes.size == 0:
                        processed_frames.append(frames[i])
                        continue

                    # Visualize detections
                    fig, ax = plt.subplots(
                        figsize=(image_size / 100, image_size / 100), dpi=100
                    )
                    ax.set_frame_on(False)
                    ax.set_axis_off()
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    self._visualize_image(
                        batch_input[i].cpu(), nms_boxes, nms_classes, nms_probs, ax=ax
                    )

                    # Convert plot to frame
                    fig.canvas.draw()
                    plotted_frame = np.array(fig.canvas.renderer.buffer_rgba())[
                        :, :, :3
                    ]
                    plotted_frame = cv2.resize(
                        plotted_frame,
                        (video_w, video_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    processed_frames.append(plotted_frame)

                    plt.close(fig)

                # Clear batch
                frames, frame_batches = [], []

        cap.release()

        if len(processed_frames) < batch_size:
            print(f"Skipped last batch as it contains less than {batch_size} frames.")

        # Save processed video
        output_video_path = os.path.join(save_dir, "processed_video.mp4")
        os.makedirs(save_dir, exist_ok=True)
        clip = ImageSequenceClip(processed_frames, fps=original_fps)
        clip.write_videofile(output_video_path, codec="libx264")
        print(f"Saved processed video to: {output_video_path}")
