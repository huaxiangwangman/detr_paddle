"""
Various positional encodings for the transformer.
"""
from x2paddle import torch2paddle
import math
import paddle
from paddle import nn
from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False,
        scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = paddle.arange(self.num_pos_feats, dtype='float32'
            ).requires_grad_(False)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = paddle.stacks(('pos_x[:, :, :, 0::2].sin()\n',
            'pos_x[:, :, :, 1::2].cos()\n'), axis=4).flatten(3)
        pos_y = paddle.stacks(('pos_y[:, :, :, 0::2].sin()\n',
            'pos_y[:, :, :, 1::2].cos()\n'), axis=4).flatten(3)
        pos = torch2paddle.concat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Layer):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        from paddle.nn.initializer import Uniform
        uniform_ = Uniform()
        self.row_embed.weight = uniform_(self.row_embed.weight)
        self.col_embed.weight = uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = paddle.arange(w).requires_grad_(False)
        j = paddle.arange(h).requires_grad_(False)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch2paddle.concat([x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1)], dim=-1).permute(2, 0, 1
            ).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f'not supported {args.position_embedding}')
    return position_embedding
