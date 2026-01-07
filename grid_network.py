import numpy as np
import torch
import torch.nn as nn

from rnn_topoloss import rnn_laplacian_pyramid_loss, rnn_decomposed_topographic_loss


class GridNetwork(nn.Module):
    """
    RNN path integrator to develop grid cells in the hidden layer

    Predicts place cell activations from velociy inputs
    """
    
    def __init__(self, Np=512, Ng=4096, weight_decay=1e-4, activation='relu'):
        super().__init__()
        
        self.Np = Np
        self.Ng = Ng
        self.weight_decay = weight_decay
        
        self.encoder = nn.Linear(Np, Ng, bias=False)
        
        self.RNN = nn.RNN(
            input_size=2,
            hidden_size=Ng,
            nonlinearity=activation,
            bias=False,
            batch_first=False # Input shape: (seq_len, batch, features)
        )
        
        # Maps grid activations to place cell predictions
        self.decoder = nn.Linear(Ng, Np, bias=False)
        
        # Softmax for converting logits to probabilities
        self.softmax = nn.Softmax(dim=-1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.decoder.weight)
    
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
    
    def compute_loss(self, velocity, init_pc, target_pc):
        """
        Computes loss with weight regularization

        Returns loss with cross-entropy + weight regularization and metrics dict with 'ce_loss', 'reg_loss', 'total_loss'
        """
        # Get predictions
        logits, _ = self.forward(velocity, init_pc)
        
        # Cross-entropy loss = -sum(y * log(softmax(pred)))
        yhat = self.softmax(logits)
        ce_loss = -(target_pc * torch.log(yhat + 1e-10)).sum(dim=-1).mean()
        
        # Weight regularlization
        reg_loss = self.weight_decay * (self.RNN.weight_hh_l0 ** 2).sum()
        
        total_loss = ce_loss + reg_loss
        
        metrics = {
            'ce_loss': ce_loss.item(),
            'reg_loss': reg_loss.item(), 
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics
    
    def compute_topographic_loss(self, velocity, init_pc, target_pc):
        """
        Computes loss with weight regularization + topoloss

        Returns loss with cross-entropy + weight regularization + topoloss and metrics dict with 'ce_loss', 'reg_loss', 'total_loss', 'topo_loss'
        """
        # Get predictions
        logits, _ = self.forward(velocity, init_pc)
        
        # Cross-entropy loss = -sum(y * log(softmax(pred)))
        yhat = self.softmax(logits)
        ce_loss = -(target_pc * torch.log(yhat + 1e-10)).sum(dim=-1).mean()
        
        # Weight regularlization
        reg_loss = self.weight_decay * (self.RNN.weight_hh_l0 ** 2).sum()

        # Topoloss
        topo_loss = rnn_laplacian_pyramid_loss(rnn_layer=self.RNN)
        
        total_loss = ce_loss + reg_loss + topo_loss
        
        metrics = {
            'ce_loss': ce_loss.item(),
            'reg_loss': reg_loss.item(), 
            'total_loss': total_loss.item(),
            'topo_loss': topo_loss.item()
        }
        
        return total_loss, metrics
    
    def compute_dtl_loss(self, velocity, init_pc, target_pc, scale_weight=1.0, phase_weight=0.1):
        """
        Computes loss with Decomposed Topographic Loss (DTL)
        
        DTL creates biologically-accurate grid cell topography:
        - Scale topography: nearby neurons have similar grid spacing
        - Phase diversity: nearby neurons have different grid offsets
        
        This matches the modular organization in entorhinal cortex where
        grid cells within a module share scale but have random phases.
        
        Args:
            velocity: Velocity inputs (seq_len, batch, 2)
            init_pc: Initial place cell activations (batch, Np)
            target_pc: Target place cell activations (seq_len, batch, Np)
            scale_weight: Weight for scale topography loss (λ_scale)
            phase_weight: Weight for phase diversity loss (λ_phase)
        
        Returns:
            total_loss: Combined loss value
            metrics: Dict with loss components
        """
        # Get predictions
        logits, _ = self.forward(velocity, init_pc)
        
        # Cross-entropy loss
        yhat = self.softmax(logits)
        ce_loss = -(target_pc * torch.log(yhat + 1e-10)).sum(dim=-1).mean()
        
        # Weight regularization
        reg_loss = self.weight_decay * (self.RNN.weight_hh_l0 ** 2).sum()
        
        # Decomposed Topographic Loss
        dtl_loss, dtl_metrics = rnn_decomposed_topographic_loss(
            rnn_layer=self.RNN,
            scale_weight=scale_weight,
            phase_weight=phase_weight,
        )
        
        total_loss = ce_loss + reg_loss + dtl_loss
        
        metrics = {
            'ce_loss': ce_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item(),
            'dtl_loss': dtl_loss.item(),
            'scale_loss': dtl_metrics['scale_loss'],
            'phase_loss': dtl_metrics['phase_loss'],
        }
        
        return total_loss, metrics
    
    def decode_position(self, place_logits, place_cell_centers, k=3):
        """
        Decodes position from top k place cell predictions
        
        Returns predicted positions of size (*, 2)            
        """
        _, idxs = torch.topk(place_logits, k=k, dim=-1) # (*, k)
        pred_pos = place_cell_centers[idxs].mean(dim=-2) # (*, 2)
        return pred_pos

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
