import torch
import torch.nn as nn
from einops import rearrange
from topoloss import laplacian_pyramid_loss


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
        # Shape: (hidden_size, hidden_size) for RNN
        weights = rnn_layer.weight_hh_l0
    elif weight_type == "input":
        weights = rnn_layer.weight_ih_l0
    else:
        raise ValueError("weight_type must be 'recurrent' or 'input'")

    # Check dimensions
    hidden_dim = rnn_layer.hidden_size
    assert map_h * map_w == hidden_dim, \
        f"Map dimensions ({map_h}x{map_w}={map_h*map_w}) must match hidden size ({hidden_dim})"

    # Loss expects (height, width, embedding_dim) but RNN weights are (output_dim, input_dim)
    # View 'output_dim' as the neurons on the sheet.
    if isinstance(rnn_layer, (nn.LSTM, nn.GRU)):
        # Use row index as neuron index
        num_gates = 4 if isinstance(rnn_layer, nn.LSTM) else 3
        
        # Reshape to (gates, hidden, input_dim)
        w_reshaped = rearrange(weights, '(g h) i -> g h i', g=num_gates, h=hidden_dim)
        
        # Rearrange to (h, (g * i))
        w_sheet_flat = rearrange(w_reshaped, 'g h i -> h (g i)')
    else:
        # Vanilla RNN
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