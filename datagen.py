import numpy as np
import torch

from trajectory_generator import TrajectoryGenerator, PlaceCells


class GridCellDataGenerator:
    """
    Generates batches of trajectories for training, batches contain:
        Velocity inputs
        Initial place cell activation
        Target place cell activation
        Ground truth positions (used in analysis)
    """

    def __init__(
        self,
        n_place_cells: int = 512,
        box_size: float = 2.2,
        dt: float = 0.02,
        place_cell_width: float = 0.12,
        surround_scale: float = 2.0,
        surround_amplitude: float = 0.5, 
        DoG: bool = True,
        sequence_length: int = 20,
        batch_size: int = 200,
        periodic: bool = False,
        device: str = 'cpu'
    ):
        self.n_place_cells = n_place_cells
        self.box_size = box_size
        self.dt = dt
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        
        self.trajectory_gen = TrajectoryGenerator(
            box_width=box_size,
            box_height=box_size,
            sequence_length=sequence_length,
            dt=dt,
            periodic=periodic
        )
        
        self.place_cells = PlaceCells(
            num_cells=n_place_cells,
            box_width=box_size,
            box_height=box_size,
            sigma=place_cell_width,
            surround_scale=surround_scale,
            surround_amplitude=surround_amplitude,
            periodic=periodic,
            DoG=DoG,
            device=device
        )
        
        print(f"Trajectory Generator initialized:")
        print(f"Mean speed: {self.trajectory_gen.mean_speed:.3f} m/s")
        print(f"Mean ego_v (displacement/step): {self.trajectory_gen.mean_ego_v:.4f} m")
        print(f"Expected velocity input magnitude: ~{self.trajectory_gen.mean_ego_v:.4f}")
        
    def generate_batch(
        self,
        batch_size: int = None,
        seq_length: int = None
    ):
        """
        Generate a batch of training data

        Returns:
            Velocity with (seq_length, batch_size, 2) shape
            Initial place cell activations (batch_size, n_place_cells)
            Target place cell activations (seq_length, batch_size, n_place_cells)
            Ground truth positions (seq_length, batch_size, 2)
        """
        if batch_size is None:
            batch_size = self.batch_size
        if seq_length is None:
            seq_length = self.sequence_length
            
        original_seq_len = self.trajectory_gen.sequence_length
        if seq_length != original_seq_len:
            self.trajectory_gen.sequence_length = seq_length
        
        # Generate trajectory batch
        traj = self.trajectory_gen.generate_trajectory(batch_size)
        
        if seq_length != original_seq_len:
            self.trajectory_gen.sequence_length = original_seq_len
        
        # Calculate allocentric velocity: [ego_v * cos(hd), ego_v * sin(hd)]
        # ego_v is displacement per step (meters), not velocity
        v = np.stack([
            traj['ego_v'] * np.cos(traj['target_hd']),
            traj['ego_v'] * np.sin(traj['target_hd'])
        ], axis=-1) # (batch, seq, 2)
        
        # Transpose to (seq, batch, 2)
        velocity = torch.tensor(
            v, dtype=torch.float32, device=self.device
        ).permute(1, 0, 2)
        
        init_pos = torch.tensor(
            np.stack([traj['init_x'], traj['init_y']], axis=-1),
            dtype=torch.float32,
            device=self.device
        ) # (batch, 2)
        
        target_pos = torch.tensor(
            np.stack([traj['target_x'], traj['target_y']], axis=-1),
            dtype=torch.float32,
            device=self.device
        ).permute(1, 0, 2) # (seq, batch, 2)
        
        init_pc = self.place_cells.get_activation(init_pos)  # (batch, n_cells)
        
        # Reshape from (seq, batch, 2) to (seq*batch, 2)
        target_pos_flat = target_pos.reshape(-1, 2)
        target_pc_flat = self.place_cells.get_activation(target_pos_flat)
        target_pc = target_pc_flat.reshape(seq_length, batch_size, -1)
        
        return velocity, init_pc, target_pc, target_pos
    
    def compute_position_error(self, pred_logits, true_pos, k=3):
        """
        Calculates position decoding error in centimeters with top-k decoding

            pred_logits: predicted logits with size (seq, batch, n_place_cells)
            true_pos: ground truth positions with size (seq, batch, 2)
            k: number of top place cells to average (default is 3)
        
        Returns mean position error in cm
        """
        with torch.no_grad():
            _, idxs = torch.topk(pred_logits, k=k, dim=-1) # (seq, batch, k)
            pred_pos = self.place_cells.us[idxs].mean(dim=-2) # (seq, batch, 2)
            
            # Compute euclidean distance
            error_m = torch.sqrt(((pred_pos - true_pos) ** 2).sum(dim=-1)).mean()
            
            # Convert to centimeters
            return error_m.item() * 100
