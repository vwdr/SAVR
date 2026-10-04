"""Downstream-only current-frame action correction; not yet a trained policy.

Inputs are detached, actual compressed-policy features and all current projected
visual patches. No cache ages, reconstructed K/V, residual bounds or routing.
"""
import torch
from torch import nn


class CorrectionBlock(nn.Module):
    def __init__(self, width, heads, visual):
        super().__init__()
        self.visual = visual
        if visual:
            self.query_norm = nn.LayerNorm(width)
            self.context_norm = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4*width),
                                 nn.GELU(), nn.Linear(4*width, width))

    def forward(self, query, context):
        if self.visual:
            context = self.context_norm(context)
            query = query + self.attention(self.query_norm(query), context, context,
                                           need_weights=False)[0]
        return query + self.ffn(query)


class CurrentFrameCorrector(nn.Module):
    """Zero-initialized normalized-action residual, with an action-only ablation.

    Default: eight action queries, 512 ordered current patches, width 256,
    two cross-attention blocks, eight heads. The input dimension is adjustable
    only to permit inexpensive CPU structural tests; production uses 4096.
    All learned weights are float32 initially. Upstream tensors are detached and
    cloned, including inference-mode tensors that autograd cannot otherwise save.
    Feature extraction, projection and these copies must be included in timing.
    """
    def __init__(self, input_dim=4096, width=256, heads=8, blocks=2, use_visual=True):
        super().__init__()
        if width % heads or min(input_dim,width,heads,blocks) < 1:
            raise ValueError("invalid corrector dimensions")
        self.input_dim, self.use_visual = input_dim, use_visual
        self.z_projection = nn.Linear(input_dim, width)
        self.action_projection = nn.Linear(7, width)
        self.state_projection = nn.Linear(8, width)
        self.instruction_projection = nn.Linear(input_dim, width)
        self.step_embedding = nn.Embedding(8, width)
        self.fusion = nn.Linear(5*width, width)
        if use_visual:
            self.visual_projection = nn.Linear(input_dim, width)
            self.camera_embedding = nn.Embedding(2, width)
            self.row_embedding = nn.Embedding(16, width)
            self.column_embedding = nn.Embedding(16, width)
        self.blocks = nn.ModuleList([CorrectionBlock(width,heads,use_visual) for _ in range(blocks)])
        self.output = nn.Linear(width,7)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _input(self, value, shape):
        if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or not value.is_floating_point():
            raise ValueError("feature shape/type differs")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite features")
        return value.detach().to(device=self.output.weight.device,dtype=self.output.weight.dtype).clone()

    def forward(self, *, action_features, base_actions, current_visual, state, instruction):
        if base_actions.ndim != 3:
            raise ValueError("batch, eight steps and seven actions required")
        b=base_actions.shape[0]
        if b<1: raise ValueError("empty batch")
        base=self._input(base_actions,(b,8,7))
        z=self._input(action_features,(b,8,self.input_dim))
        proprio=self._input(state,(b,8))
        language=self._input(instruction,(b,self.input_dim))
        steps=torch.arange(8,device=base.device)
        query=self.fusion(torch.cat((self.z_projection(z),self.action_projection(base),
            self.state_projection(proprio)[:,None,:].expand(-1,8,-1),
            self.instruction_projection(language)[:,None,:].expand(-1,8,-1),
            self.step_embedding(steps)[None,:,:].expand(b,-1,-1)),dim=-1))
        context=None
        if self.use_visual:
            visual=self._input(current_visual,(b,512,self.input_dim))
            ids=torch.arange(512,device=base.device)
            context=(self.visual_projection(visual)+self.camera_embedding(ids//256)
                     +self.row_embedding((ids%256)//16)+self.column_embedding(ids%16))
        for block in self.blocks:
            query=block(query,context)
        return base+self.output(query)


def teacher_imitation_loss(predicted, dense_target):
    """Mean L1 over normalized 8x7 chunks; teacher labels never receive gradients."""
    if predicted.shape!=dense_target.shape or predicted.shape[-2:]!=(8,7):
        raise ValueError("teacher chunk alignment differs")
    target=dense_target.detach().to(predicted).clone()
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("nonfinite training values")
    return (predicted-target).abs().mean()
