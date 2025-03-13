import torch
from torch import nn


class ConditionalDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable anchor positions (x,y pairs)
        self.spatial_2d_coords = nn.Linear(d_model, 2)

        # Displacement FFN for spatial queries (calculated from the object queries)
        self.displacement_ffn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

        # Learnable diagonal transformation (λq)
        self.lambda_q = nn.Parameter(torch.ones(d_model))

    def positional_encoding(self, ref_points):
        """
        Applies sinusoidal encoding to reference points.
        ref_points: (batch_size, num_queries, 2) -> (batch_size, num_queries, d_model)
        """
        # Normalize using sigmoid
        ref_points = torch.sigmoid(ref_points)  # Normalize to [0,1]

        # Get half the dimension size for each axis..
        half_dim = self.lambda_q.shape[0] // 2

        # Compute sinusoidal embeddings (like Transformer positional encoding)
        dim_t = torch.arange(half_dim, dtype=torch.float32, device=ref_points.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / half_dim)

        pos_x = ref_points[..., 0, None] / dim_t
        pos_y = ref_points[..., 1, None] / dim_t

        pos_x = torch.cat(
            [torch.sin(pos_x[:, :, ::2]), torch.cos(pos_x[:, :, 1::2])], dim=-1
        )
        pos_y = torch.cat(
            [torch.sin(pos_y[:, :, ::2]), torch.cos(pos_y[:, :, 1::2])], dim=-1
        )

        return torch.cat([pos_x, pos_y], dim=-1)  # (batch_size, num_queries, d_model)

    def forward(self, decoder_embeddings, object_queries, encoder_output_feats):

        # Get displacements from the decoder embeddings
        learnable_displacements = self.displacement_ffn(decoder_embeddings)

        # Add object queries to decoder embeddings to create KQ pairs
        self_attn_key = decoder_embeddings + object_queries
        self_attn_query = decoder_embeddings + object_queries

        # Self-attention
        self_attn_output = self.self_attn(
            self_attn_query, self_attn_key, decoder_embeddings
        )[0]
        decoder_embeddings = self.norm1(
            decoder_embeddings + self.dropout(self_attn_output)
        )

        # Get the 2D reference coordinates embeddings
        ref_coords_2d = self.spatial_2d_coords(
            object_queries
        )  # (batch_size, num_queries, 2)

        # Convert to sinusoidal embeddings
        ref_points_embed = self.positional_encoding(
            ref_coords_2d
        )  # (batch_size, num_queries, d_model)

        # **Compute spatial query pq = T(f) ⊙ p_s (element-wise multiplication)**
        spatial_query = learnable_displacements * (
            self.lambda_q * ref_points_embed
        )  # (batch_size, num_queries, d_model)

        # Conditional cross-attention
        cross_query = decoder_embeddings + spatial_query
        cross_attn_output = self.cross_attn(
            cross_query, encoder_output_feats, encoder_output_feats
        )[0]
        decoder_embeddings = self.norm2(
            decoder_embeddings + self.dropout(cross_attn_output)
        )

        # Feedforward
        ffn_out = self.ffn(decoder_embeddings)
        decoder_embeddings = self.norm3(decoder_embeddings + self.dropout(ffn_out))

        # Convert the (cx,cy) coordinates to (cx,cy, 0, 0)
        reference_point = torch.cat(
            [ref_coords_2d, torch.zeros_like(ref_coords_2d)], dim=-1
        )

        return decoder_embeddings, reference_point
