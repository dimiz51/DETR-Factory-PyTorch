from einops import rearrange
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn
import torch


def get_hook(outs, name):
    """This will store a layer's output in a dictionary using a forward hook

    Args:
        outs (dict): Dictionary to store the layer's output
        name (str): Name of the layer

    Returns:
        hook (function): A forward hook

    """

    def hook(self, input, output):
        outs[name] = output

    return hook


class DETR(nn.Module):
    """Detection Transformer (DETR) model with a ResNet50 backbone.

    Paper: https://arxiv.org/abs/2005.12872

    Args:
        d_model (int, optional): Embedding dimension. Defaults to 256.
        n_classes (int, optional): Number of classes. Defaults to 92.
        n_tokens (int, optional): Number of tokens. Defaults to 225.
        n_layers (int, optional): Number of layers. Defaults to 6.
        n_heads (int, optional): Number of heads. Defaults to 8.
        n_queries (int, optional): Number of queries/max objects. Defaults to 100.

    Returns:
        DETR: DETR model
    """

    def __init__(
        self,
        d_model=256,
        n_classes=92,
        n_tokens=225,
        n_layers=6,
        n_heads=8,
        n_queries=100,
    ):
        super().__init__()

        self.backbone = create_feature_extractor(
            torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True),
            return_nodes={"layer4": "layer4"},
        )

        self.conv1x1 = nn.Conv2d(2048, d_model, kernel_size=1, stride=1)

        self.pe_encoder = nn.Parameter(
            torch.rand((1, n_tokens, d_model)), requires_grad=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.queries = nn.Parameter(
            torch.rand((1, n_queries, d_model)), requires_grad=True
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )

        # Each of the decoder's outputs will be passed through
        # linear layers for prediction of boxes/classes.
        self.linear_class = nn.Linear(d_model, n_classes)
        self.linear_bbox = nn.Linear(d_model, 4)

        # Add hooks to get intermediate outcomes
        self.decoder_outs = {}
        for i, L in enumerate(self.transformer_decoder.layers):
            name = f"layer_{i}"
            L.register_forward_hook(get_hook(self.decoder_outs, name))

    def forward(self, x):
        # Pass inputs through the CNN backbone...
        tokens = self.backbone(x)["layer4"]

        # Pass outputs from the backbone through a simple conv...
        tokens = self.conv1x1(tokens)

        # Re-order in patches format
        tokens = rearrange(tokens, "b c h w -> b (h w) c")

        # Pass encoded patches through encoder...
        out_encoder = self.transformer_encoder((tokens + self.pe_encoder))

        # We expand so each image of each batch get's it's own copy of the
        # query embeddings. So from (1, 100, 256) to (4, 100, 256) for example
        # for batch size=4, with 100 queries of embedding dimension 256.
        # Then we pass through the decoder...
        out_decoder = self.transformer_decoder(
            self.queries.repeat(out_encoder.shape[0], 1, 1), out_encoder
        )

        # Compute outcomes for all intermediate
        # decoder's layers...
        # NOTE: The hook we registered previously is called during each
        # forward pass and will store the layer's output in 'self.decoder_outs',
        # where we can easily access them and then pass them through
        # linear layers for prediction.
        outs = {}
        for n, o in self.decoder_outs.items():
            outs[n] = {"cl": self.linear_class(o), "bbox": self.linear_bbox(o)}

        return outs
