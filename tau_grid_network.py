import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tau_rnn import TauRNN, tau_topographic_loss


class TauGridNetwork(nn.Module):
    """
    RNN integrator with tau- (time-constant)-based dynamics for path integration
    
        Np: Number of place cells (output dimension)
        Ng: Number of grid cells (hidden units, must be perfect square)
        weight_decay: L2 regularization on weights
        activation: Activation function ('relu' or 'tanh')
        tau_min: Minimum time constant
        tau_max: Maximum time constant
        tau_init: Initialization strategy ('gradient', 'random', 'uniform', 'modules', 'radial')
        learnable_tau: Whether tau should be learnable (True) or fixed (False)
        n_modules: Number of discrete modules (for 'modules' init)
    """

    def __init__(
        self,
        Np = 512,
        Ng = 4096,
        weight_decay = 1e-4,
        activation = 'relu',
        tau_min = 1.0,
        tau_max = 50.0,
        tau_init = 'gradient',
        learnable_tau = True,
        n_modules = 4,
    ):
        super().__init__()

        self.Np = Np
        self.Ng = Ng
        self.weight_decay = weight_decay
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.encoder = nn.Linear(Np, Ng, bias=False)

        # Tau-based RNN
        self.RNN = TauRNN(
            input_size=2,
            hidden_size=Ng,
            activation=activation,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_init=tau_init,
            learnable_tau=learnable_tau,
            n_modules=n_modules,
            bias=False,
        )

        # Maps grid activations to place cell predictions
        self.decoder = nn.Linear(Ng, Np, bias=False)

        # Softmax for converting logits to probabilities
        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        """Initialize encoder and decoder weights"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    @property
    def tau(self):
        """Get tau values from the RNN"""
        return self.RNN.tau

    def get_tau_2d(self):
        """Get tau reshaped as 2D cortical sheet"""
        return self.RNN.get_tau_2d()

    def g(self, velocity, init_pc):
        """
        Calculate grid cell activations from velocity input
                
        Returns grid cell activations of size (seq_len, batch, Ng)
        """
        h0 = self.encoder(init_pc).unsqueeze(0) # (1, batch, Ng)

        g, _ = self.RNN(velocity, h0) # (seq_len, batch, Ng)

        return g

    def forward(self, velocity, init_pc):
        """
        Returns predicted place_logits of size (seq_len, batch, Np) and grid cell activations of size (seq_len, batch, Ng)
        """
        grid_activations = self.g(velocity, init_pc)
        place_logits = self.decoder(grid_activations)
        return place_logits, grid_activations

    def compute_tau_topographic_loss(
        self,
        velocity,
        init_pc,
        target_pc,
        tau_smoothness_weight = 1.0,
        tau_variance_weight = 0.1,
    ):
        """
        Compute loss with topographic regularization on tau (use when tau is learnable and when it should self organize)
        
            tau_smoothness_weight: Weight for local tau smoothness
            tau_variance_weight: Weight for tau variance (diversity)
        """
        logits, _ = self.forward(velocity, init_pc)

        # Cross-entropy loss
        yhat = self.softmax(logits)
        ce_loss = -(target_pc * torch.log(yhat + 1e-10)).sum(dim=-1).mean()

        # Weight regularization
        reg_loss = self.weight_decay * (self.RNN.weight_hh_l0 ** 2).sum()

        # Tau topographic loss
        tau_loss, tau_metrics = tau_topographic_loss(
            self.RNN,
            smoothness_weight=tau_smoothness_weight,
            variance_weight=tau_variance_weight,
        )

        total_loss = ce_loss + reg_loss + tau_loss

        metrics = {
            'ce_loss': ce_loss.item(),
            'reg_loss': reg_loss.item(),
            'tau_loss': tau_loss.item(),
            'total_loss': total_loss.item(),
            **tau_metrics,
        }

        return total_loss, metrics

    def decode_position(self, place_logits, place_cell_centers, k = 3):
        """
        Decodes position from top k place cell predictions
        
        Returns predicted positions of size (*, 2)            
        """
        _, idxs = torch.topk(place_logits, k=k, dim=-1)
        pred_pos = place_cell_centers[idxs].mean(dim=-2)
        return pred_pos