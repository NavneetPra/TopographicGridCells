from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TauRNN(nn.Module):
    """
    Custom RNN with learnable tau per neuron
    
    The tau parameter determines the integration time constant:
    - Large tau -> slow integration
    - Small tau -> fast integration
    
    Tau can be initialized:
    - 'gradient': Smooth gradient 
    - 'random': Random values 
    - 'uniform': Same value for all neurons
    - 'modules': Discrete modules with different tau values
    
        input_size: Dimension of input features
        hidden_size: Number of hidden units (should be perfect square)
        activation: Activation function ('relu' or 'tanh')
        tau_min: Minimum time constant
        tau_max: Maximum time constant
        tau_init: Initialization strategy for tau ('gradient', 'random', 'uniform', 'modules')
        learnable_tau: Whether tau should be learnable
        n_modules: Number of discrete modules (only used if tau_init='modules')
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        activation = 'relu',
        tau_min = 1.0,
        tau_max = 50.0,
        tau_init = 'gradient',
        learnable_tau = True,
        n_modules = 4,
        bias = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_init_mode = tau_init
        self.learnable_tau = learnable_tau
        self.n_modules = n_modules

        # Compute side length for cortical sheet
        self.side = int(np.sqrt(hidden_size))
        assert self.side * self.side == hidden_size, \
            f"hidden_size ({hidden_size}) must be a perfect square"

        # Standard RNN weights (matching PyTorch RNN)
        self.weight_ih_l0 = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(hidden_size, hidden_size))

        if bias:
            self.bias_ih_l0 = nn.Parameter(torch.zeros(hidden_size))
            self.bias_hh_l0 = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_buffer('bias_ih_l0', None)
            self.register_buffer('bias_hh_l0', None)

        tau_init_values = self._init_tau(tau_init, tau_min, tau_max, n_modules)

        if learnable_tau:
            # Use log parameterization for stability
            self.log_tau = nn.Parameter(torch.log(tau_init_values))
        else:
            self.register_buffer('log_tau', torch.log(tau_init_values))

        # Activation function
        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._activation_name = activation

        self._init_weights()

    def _init_tau(self, mode, tau_min, tau_max, n_modules):
        """Initialize tau values based on specified strategy"""

        if mode == 'gradient':
            # Smooth gradient from tau_min at top to tau_max at bottom
            tau_values = torch.linspace(tau_min, tau_max, self.side)
            tau_init = tau_values.unsqueeze(1).expand(self.side, self.side).flatten()
        elif mode == 'radial':
            # Gradient with tau_min at center, tau_max at edges
            y_coords = torch.linspace(-1, 1, self.side)
            x_coords = torch.linspace(-1, 1, self.side)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            distances = torch.sqrt(xx**2 + yy**2)
            distances = distances / distances.max() 
            tau_init = tau_min + (tau_max - tau_min) * distances.flatten()
        elif mode == 'modules':
            # Discrete modules with different tau values (horizontal stripes)
            module_idx = torch.arange(self.side) // (self.side // n_modules)
            module_idx = module_idx.clamp(max=n_modules - 1)
            tau_per_module = torch.linspace(tau_min, tau_max, n_modules)
            tau_row = tau_per_module[module_idx]
            tau_init = tau_row.unsqueeze(1).expand(self.side, self.side).flatten()
        elif mode == 'random':
            # Random initialization 
            tau_init = torch.rand(self.hidden_size) * (tau_max - tau_min) + tau_min
        elif mode == 'uniform':
            # Same tau for all neurons
            tau_init = torch.full((self.hidden_size,), (tau_min + tau_max) / 2)
        else:
            raise ValueError(f"Unknown tau_init mode: {mode}")

        return tau_init.float()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.weight_ih_l0)
        nn.init.xavier_uniform_(self.weight_hh_l0)

    @property
    def tau(self):
        """Get tau values clamped to valid range"""
        return torch.exp(self.log_tau).clamp(self.tau_min, self.tau_max)

    @property
    def alpha(self):
        """Get integration rate (1/tau)"""
        return 1.0 / self.tau

    def get_tau_2d(self):
        """Get tau reshaped as 2D cortical sheet"""
        return self.tau.view(self.side, self.side)

    def forward(self, input: torch.Tensor, h0: torch.Tensor = None):
        """
        Forward pass through the tau-based RNN
        """
        seq_len, batch_size, _ = input.shape
        device = input.device

        # Handle initial hidden state
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
        elif h0.dim() == 3:
            h = h0.squeeze(0) 
        else:
            h = h0

        outputs = []

        # Get integration rate (per neuron)
        alpha = self.alpha.to(device) # (hidden_size,)

        for t in range(seq_len):
            # Standard RNN pre-activation
            pre_act = F.linear(input[t], self.weight_ih_l0, self.bias_ih_l0) + \
                      F.linear(h, self.weight_hh_l0, self.bias_hh_l0)

            # Apply activation
            post_act = self.activation(pre_act)

            # Leaky integration with per-neuron time constant
            h = (1 - alpha) * h + alpha * post_act

            outputs.append(h)

        # Stack outputs (seq_len, batch_size, hidden_size)
        output = torch.stack(outputs, dim=0)

        # Final hidden state with num_layers dimension (1, batch_size, hidden_size)
        h_n = h.unsqueeze(0)

        return output, h_n

    def extra_repr(self):
        return (f'input_size={self.input_size}, hidden_size={self.hidden_size}, '
                f'activation={self._activation_name}, tau_range=[{self.tau_min}, {self.tau_max}], '
                f'tau_init={self.tau_init_mode}, learnable_tau={self.learnable_tau}')

def tau_topographic_loss(
    tau_rnn,
    factor = 2.0,
    smoothness_weight = 1.0,
    variance_weight = 0.1,
):
    """
    Topographic loss for tau values with spatial smoothness and global variance
    
        tau_rnn: The TauRNN module
        factor: Downsampling factor for Laplacian pyramid
        smoothness_weight: Weight for local smoothness loss
        variance_weight: Weight for variance (diversity) loss (negative = maximize)
    """
    tau = tau_rnn.tau
    side = tau_rnn.side

    # Reshape to 2D
    tau_2d = tau.view(1, 1, side, side)

    # Smoothness
    downscaled = F.interpolate(tau_2d, scale_factor=1/factor, mode='bilinear', align_corners=False)
    upscaled = F.interpolate(downscaled, size=(side, side), mode='bilinear', align_corners=False)

    smoothness_loss = ((tau_2d - upscaled) ** 2).mean()

    # Variance
    variance_loss = -tau.var()

    total_loss = smoothness_weight * smoothness_loss + variance_weight * variance_loss

    metrics = {
        'tau_smoothness': smoothness_loss.item(),
        'tau_variance': tau.var().item(),
        'tau_mean': tau.mean().item(),
        'tau_min': tau.min().item(),
        'tau_max': tau.max().item(),
        'total_tau_loss': total_loss.item(),
    }

    return total_loss, metrics

def tau_topographic_loss_v2(
    tau_rnn,
    factor = 2.0,
    smoothness_weight = 1.0,
    variance_weight = 0.1,
):
    """
    Topographic loss for tau values with spatial smoothness and global variance
    
        tau_rnn: The TauRNN module
        factor: Downsampling factor for Laplacian pyramid
        smoothness_weight: Weight for local smoothness loss
        variance_weight: Weight for variance loss
    """
    tau = tau_rnn.tau
    side = tau_rnn.side

    tau_2d = tau.view(1, 1, side, side)

    # Smoothness
    downscaled = F.interpolate(tau_2d, scale_factor=1/factor, mode='bilinear', align_corners=False)
    upscaled = F.interpolate(downscaled, size=(side, side), mode='bilinear', align_corners=False)

    smoothness_loss = ((tau_2d - upscaled) ** 2).mean()

    # Variance
    tau_sorted, _ = torch.sort(tau.flatten())

    # Ideal uniform distribution
    n_units = tau.numel()
    current_min = tau.min().detach()
    current_max = tau.max().detach()

    # Linear ramp
    target_dist = torch.linspace(current_min, current_max, n_units, device=tau.device)

    # Compare actual to linear ramp
    variance_loss = F.mse_loss(tau_sorted, target_dist)

    total_loss = smoothness_weight * smoothness_loss + variance_weight * variance_loss

    metrics = {
        'tau_smoothness': smoothness_loss.item(),
        'tau_variance': tau.var().item(),
        'tau_mean': tau.mean().item(),
        'tau_min': tau.min().item(),
        'tau_max': tau.max().item(),
        'total_tau_loss': total_loss.item(),
    }

    return total_loss, metrics
