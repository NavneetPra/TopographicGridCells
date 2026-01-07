import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from topoloss import laplacian_pyramid_loss


def decomposed_topographic_loss(
    weight_matrix: torch.Tensor,
    side: int = 64,
    factor: float = 2.0,
    scale_weight: float = 1.0,
    phase_weight: float = 0.1,
):
    """
    Loss handling magnitude and direction of weight vectors seperately to maximize scale topography and minimize phase topography
    """
    hidden_size = weight_matrix.shape[0]
    
    # Scale topography
    
    magnitudes = torch.norm(weight_matrix, dim=1)  
    mag_2d = magnitudes.view(1, 1, side, side) # Reshape to 2D cortical sheet
    
    # Create blurred version with downsample then upsample
    downscaled_mag = F.interpolate(mag_2d, scale_factor=1/factor, mode='bilinear', align_corners=False)
    upscaled_mag = F.interpolate(downscaled_mag, size=(side, side), mode='bilinear', align_corners=False)
    
    # Scale loss
    scale_loss = ((mag_2d - upscaled_mag) ** 2).mean()
    
    # Phase diversity
    
    norms = torch.norm(weight_matrix, dim=1, keepdim=True) + 1e-8
    directions = weight_matrix / norms 
    
    # Reshape to 2D cortical sheet
    dirs_2d = directions.view(side, side, -1).permute(2, 0, 1).unsqueeze(0)
    
    # Create blurred version of direction vectors
    downscaled_dir = F.interpolate(dirs_2d, scale_factor=1/factor, mode='bilinear', align_corners=False)
    upscaled_dir = F.interpolate(downscaled_dir, size=(side, side), mode='bilinear', align_corners=False)
    
    # Flatten for cosine similarity
    dirs_flat = dirs_2d.squeeze(0).permute(1, 2, 0).reshape(-1, hidden_size)
    upscaled_flat = upscaled_dir.squeeze(0).permute(1, 2, 0).reshape(-1, hidden_size)
    
    # Normalize blurred directions
    upscaled_norms = torch.norm(upscaled_flat, dim=1, keepdim=True) + 1e-8
    upscaled_normalized = upscaled_flat / upscaled_norms
    
    # Cosine similarity between original and blurred directions
    cos_sim = (dirs_flat * upscaled_normalized).sum(dim=1)
    
    phase_loss = cos_sim.mean()
    
    total_loss = scale_weight * scale_loss + phase_weight * phase_loss
    
    metrics = {
        'scale_loss': scale_loss.item(),
        'phase_loss': phase_loss.item(),
        'total_dtl': total_loss.item(),
    }
    
    return total_loss, metrics

def rnn_decomposed_topographic_loss(
    rnn_layer: nn.RNNBase,
    map_h: int = 64,
    map_w: int = 64,
    factor: float = 2.0,
    scale_weight: float = 1.0,
    phase_weight: float = 0.1,
):
    """
    Wrapper for decomposed_topographic_loss for RNN layers
    """
    weights = rnn_layer.weight_hh_l0
    return decomposed_topographic_loss(
        weight_matrix=weights,
        side=map_h,
        factor=factor,
        scale_weight=scale_weight,
        phase_weight=phase_weight,
    )

def rnn_laplacian_pyramid_loss(
    rnn_layer: nn.RNNBase, 
    map_h: int = 64, 
    map_w: int = 64, 
    factor_h: float = 2.0, 
    factor_w: float = 2.0, 
    weight_type: str = "recurrent",
    interpolation: str = "bilinear"
):
    """
    Adapts laplacian_pyramid_loss for RNN layers.
    """
    # Gets full concatenated vector for each neuron.
    if weight_type == "recurrent":
        # Shape is (hidden_size, hidden_size) for RNN
        weights = rnn_layer.weight_hh_l0
    elif weight_type == "input":
        weights = rnn_layer.weight_ih_l0
    else:
        raise ValueError("weight_type must be 'recurrent' or 'input'")

    # Check dimensions
    hidden_dim = rnn_layer.hidden_size
    assert map_h * map_w == hidden_dim, \
        f"Map dimensions ({map_h}x{map_w}={map_h*map_w}) must match hidden size ({hidden_dim})"

    w_sheet_flat = weights

    # Form sheet (height, width, features)
    cortical_sheet = rearrange(w_sheet_flat, '(h w) e -> h w e', h=map_h, w=map_w)

    # Call original loss function
    return laplacian_pyramid_loss(
        cortical_sheet=cortical_sheet,
        factor_w=factor_w,
        factor_h=factor_h,
        interpolation=interpolation
    )