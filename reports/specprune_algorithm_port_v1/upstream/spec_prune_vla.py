import numpy as np
import torch
from skimage.util import view_as_blocks


@torch.no_grad()
def patchify(image, patch_size=14):
    """
    Converts an image into non-overlapping patches (PyTorch GPU version).
    Args:
        image: Input image (PIL or numpy array).
        patch_size: Size of each patch (default=14).
    Returns:
        Patches as a PyTorch tensor on GPU.
    """
    image = np.array(image)
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0, "Image dimensions must be divisible by patch size."

    if image.ndim == 3:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size, image.shape[2]))
    else:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size))

    patches = blocks.reshape(-1, patch_size, patch_size, image.shape[2]) if image.ndim == 3 else blocks.reshape(-1, patch_size, patch_size)
    return patches


def calculate_patch_similarity(patch1, patch2):
    """Computes cosine similarity between two sets of patches."""
    flat1 = patch1.reshape(len(patch1), -1).astype(np.float32)
    flat2 = patch2.reshape(len(patch2), -1).astype(np.float32)

    norm1 = np.linalg.norm(flat1, axis=1)
    norm2 = np.linalg.norm(flat2, axis=1)

    dot = np.sum(flat1 * flat2, axis=1)
    cosine_sim = dot / (norm1 * norm2 + 1e-8)

    return cosine_sim


def preprocess_image_for_patchify(image, patch_size=14):
    """Center-crop an image so its height and width are divisible by patch_size."""
    image = np.array(image)
    h, w = image.shape[:2]
    target_h = (h // patch_size) * patch_size
    target_w = (w // patch_size) * patch_size
    if target_h == 224 and target_w == 224:
        return image
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return image[start_h:start_h + target_h, start_w:start_w + target_w]


def get_similarity_indices(img_0, img_1, top_k, patch_size=14, sim_threshold=0.996, view="primary"):
    """Return low-change patch token ids between the previous and current frames.

    These ids do not define static reuse. They mark visually stable patches so
    the early local pruning stage can protect the complementary dynamic region.
    """
    img_0 = preprocess_image_for_patchify(img_0)
    img_1 = preprocess_image_for_patchify(img_1)
    patches1 = patchify(img_0, patch_size)
    patches2 = patchify(img_1, patch_size)

    similarity = calculate_patch_similarity(patches1, patches2)

    grid_h = img_0.shape[0] // patch_size
    grid_w = img_0.shape[1] // patch_size
    flat_indices = np.where(similarity >= sim_threshold)[0]

    if len(flat_indices) == 0:
        return np.array([], dtype=int)

    k = min(top_k, len(flat_indices))
    top_k_flat = flat_indices[np.argpartition(similarity[flat_indices], -k)[-k:]]

    row_ids, col_ids = np.unravel_index(top_k_flat, (grid_h, grid_w))
    patch_ids = row_ids * grid_w + col_ids + 1  # +1 for 1-based indexing

    if view == "wrist":
        patch_ids += 256

    return patch_ids


def vlm_layer_attn(multihead_attention, num_tokens=34, layer_indices=None, primary=True):
    # multihead_attention (tuple) --> (attention[layer_idx], cache_position[layer_idx])
    num_layers = len(multihead_attention)

    layer_indices = range(layer_indices) if layer_indices is not None else range(num_layers)

    v_token_start = 1 if primary else 257
    v_token_end = v_token_start + 256
    t_token_start = 513
    t_token_end = t_token_start + 34

    attn_dict = {}
    token_position = {}
    for layer_idx in layer_indices:
        attn_map = multihead_attention[layer_idx][0].to(torch.float32).squeeze(0).mean(dim=0)
        token_idx = multihead_attention[layer_idx][1]

        text_mask = (token_idx >= t_token_start + 9) & (token_idx < t_token_end)
        vision_mask = (token_idx >= v_token_start) & (token_idx < v_token_end)

        relation = attn_map[text_mask][:, vision_mask]
        token_position[layer_idx] = token_idx
        attn_dict[layer_idx] = relation.mean(dim=0)

    return attn_dict, token_position


def get_layer_attn_indices(multihead_attention, topk, primary=True, goal_layers=None):
    """Collect top attention patch token ids from specified goal layers.

    Returns:
        torch.Tensor: Unique top patches from goal layers, offset-adjusted for VLM token ids.
    """
    if goal_layers is None:
        goal_layers = []

    attn_dict, position = vlm_layer_attn(multihead_attention, layer_indices=None, primary=primary)

    offset = 1 if primary else 257
    total_patches = 256 + offset
    global_top = []

    for layer_idx in sorted(attn_dict.keys()):
        if layer_idx not in goal_layers:
            continue

        position_id = position[layer_idx]
        attn_scores = attn_dict[layer_idx]

        filtered_ids = (position_id[(position_id >= offset) & (position_id < total_patches)] - offset).cpu().numpy()

        full_attn = torch.zeros(256, device=attn_scores.device)
        full_attn[filtered_ids] = attn_scores
        attn = full_attn.cpu().numpy().astype(np.float32).reshape(16, 16)

        flat = [(i * 16 + j, attn[i, j]) for i in range(16) for j in range(16)]
        flat.sort(key=lambda x: x[1], reverse=True)
        global_top.extend(idx for idx, _ in flat[:topk])

    unique_global_top = sorted(set(global_top))
    return torch.tensor(unique_global_top) + offset
