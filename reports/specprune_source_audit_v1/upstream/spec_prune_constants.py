"""Hyperparameters for SpecPrune-VLA inference.

1. Static reuse keeps action-centric visual tokens selected in the previous
   control step and protects them during early pruning.
2. Local task-relevant update uses frame-difference cues plus text-guided
   attention in the first two layers to keep current dynamic visual evidence.
3. Layer-adaptive dynamic pruning updates action-aware importance scores at
   selected layers and prunes low-scoring visual tokens at target layers.
4. The action-aware controller switches to a more conservative pruning profile
   when the predicted motion indicates fine manipulation.
"""

# Visual token layout inserted after BOS: 256 primary-view patches followed by
# 256 wrist-view patches. Token ids are sequence positions in the multimodal
# LLaMA input.
NUM_VISION_TOKENS_PER_VIEW = 256
PRIMARY_VISUAL_TOKEN_START = 1
WRIST_VISUAL_TOKEN_START = PRIMARY_VISUAL_TOKEN_START + NUM_VISION_TOKENS_PER_VIEW
VISUAL_TOKEN_END = WRIST_VISUAL_TOKEN_START + NUM_VISION_TOKENS_PER_VIEW

# Dynamic pruning keeps this fraction of the current sequence at each target
# layer. Non-visual tokens are always protected.
DYNAMIC_PRUNE_RATIO = 0.9

# Scales early text-guided local selection. Values below 1 prune more
# aggressively in the first two layers.
STATIC_PRUNE_RATIO = 0.8

# First-two-layer text-guided local task-relevant token budgets.
ATTN_TOPK = int(30 * STATIC_PRUNE_RATIO)
ATTN_TOPK_WRIST = int(24 * STATIC_PRUNE_RATIO)

if STATIC_PRUNE_RATIO <= 1.0:
    ATTN_TOPK_PRECISE = int(ATTN_TOPK * STATIC_PRUNE_RATIO**2)
    ATTN_TOPK_PRECISE_WRIST = int(ATTN_TOPK_WRIST * STATIC_PRUNE_RATIO**2)
else:
    ATTN_TOPK_PRECISE = int(ATTN_TOPK * STATIC_PRUNE_RATIO)
    ATTN_TOPK_PRECISE_WRIST = int(ATTN_TOPK_WRIST * STATIC_PRUNE_RATIO)

# Frame-similarity budgets used to identify low-change patches. Their
# complement is the current local dynamic region protected in early layers.
PRIMARY_TOPK = 236
PRIMARY_TOPK_PRECISE = 240
WRIST_TOPK = 230
WRIST_TOPK_PRECISE = 236
PRIMARY_SIM_THRESHOLD = 0.986
WRIST_SIM_THRESHOLD = 0.98

# Layer-adaptive dynamic pruning schedule. Scores are updated before each target
# pruning layer, using action-token attention as the action-centric signal.
DYNAMIC_IMPORTANCE_UPDATE_LAYERS = (14, 19, 24)
DYNAMIC_PRUNE_LAYERS = (10, 15, 20, 25)
DYNAMIC_IMPORTANCE_EMA_BETA = 0.2
MIN_VISUAL_RETAIN_TOKENS = 60
DYNAMIC_PRUNE_SAFETY_MARGIN = 58

# Action-aware controller thresholds. Conservative / precise mode is enabled
# for slow or fine-grained motion and disabled for fast vertical or rotational
# motion.
VELOCITY_THRES = 0.35
ROT_THRES = 0.2
XY_THRES = 0.1
Z_THRES = 0.1
PRECISE_EXIT_Z_THRES = 0.2
PRECISE_EXIT_ROT_THRES = 0.3

# Optional layer skipping budget.
SKIP_LAYER = 3
