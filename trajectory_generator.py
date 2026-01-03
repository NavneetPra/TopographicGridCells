import numpy as np
import torch


class TrajectoryGenerator:
    """
    Generates random trajectories in a rectangular box
    
    Same as methods in https://github.com/ganguli-lab/grid-pattern-formation/tree/master
    """
    
    def __init__(
        self,
        box_width: float = 2.2,
        box_height: float = 2.2,
        sequence_length: int = 20,
        dt: float = 0.02,
        periodic: bool = False
    ):
        self.box_width = box_width
        self.box_height = box_height
        self.sequence_length = sequence_length
        self.dt = dt
        self.periodic = periodic
        
        self.sigma = 5.76 * 2 # 11.52 rad/sec rotation velocity stdev
        self.b = 0.13 * 2 * np.pi # 0.817 m/s Rayleigh scale
        self.mu = 0 # turn angle bias
        self.border_region = 0.03 # meters
        
        self.mean_speed = self.b * np.sqrt(np.pi / 2) # ~1.02 m/s
        self.mean_ego_v = self.mean_speed * self.dt # ~0.0204 m/step
    
    def avoid_wall(self, position, hd):
        """
        Returns if near wall and the turn angle to the wall
        """
        x = position[:, 0]
        y = position[:, 1]
        
        dists = [
            self.box_width / 2 - x, # right wall
            self.box_height / 2 - y, # top wall
            self.box_width / 2 + x, # left wall
            self.box_height / 2 + y # bottom wall
        ]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4) * np.pi / 2
        theta = angles[np.argmin(dists, axis=0)]
        
        hd = np.mod(hd, 2 * np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi
        
        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = np.zeros_like(hd)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (
            np.pi / 2 - np.abs(a_wall[is_near_wall])
        )
        
        return is_near_wall, turn_angle
    
    def generate_trajectory(self, batch_size: int) -> dict:
        """
        Generates random trajectory

        Returns trajectory dict with:
            init_x, init_y: initial positions with size (batch,)
            ego_v: velocities in m/step with size (batch, sequence_length)
            target_hd: head directions with size (batch, sequence_length)
            target_x, target_y: target positions with size (batch, sequence_length)
        """
        samples = self.sequence_length
        
        position = np.zeros([batch_size, samples + 2, 2])
        head_dir = np.zeros([batch_size, samples + 2])
        velocity = np.zeros([batch_size, samples + 2])
        
        # Random initial position and heading
        position[:, 0, 0] = np.random.uniform(
            -self.box_width / 2, self.box_width / 2, batch_size
        )
        position[:, 0, 1] = np.random.uniform(
            -self.box_height / 2, self.box_height / 2, batch_size
        )
        head_dir[:, 0] = np.random.uniform(0, 2 * np.pi, batch_size)
        
        # Pre-generate random values
        random_turn = np.random.normal(self.mu, self.sigma, [batch_size, samples + 1])
        random_vel = np.random.rayleigh(self.b, [batch_size, samples + 1])
        
        for t in range(samples + 1):
            # Get velocity from Rayleigh distribution
            v = random_vel[:, t]
            turn_angle = np.zeros(batch_size)
            
            if not self.periodic:
                # Wall avoidance
                is_near_wall, turn_angle = self.avoid_wall(
                    position[:, t], head_dir[:, t]
                )
                v[is_near_wall] *= 0.25  # Slow down near walls
            
            # Add random rotation (scaled by dt)
            turn_angle += self.dt * random_turn[:, t]
            
            # Store velocity (displacement per step = v * dt)
            velocity[:, t] = v * self.dt
            
            # Calculate position update
            update = velocity[:, t, None] * np.stack([
                np.cos(head_dir[:, t]),
                np.sin(head_dir[:, t])
            ], axis=-1)
            position[:, t + 1] = position[:, t] + update
            
            # Update head direction
            head_dir[:, t + 1] = head_dir[:, t] + turn_angle
        
        # Handle periodic boundaries
        if self.periodic:
            position[:, :, 0] = np.mod(
                position[:, :, 0] + self.box_width / 2, self.box_width
            ) - self.box_width / 2
            position[:, :, 1] = np.mod(
                position[:, :, 1] + self.box_height / 2, self.box_height
            ) - self.box_height / 2
        
        # Wrap head direction to [-pi, pi]
        head_dir = np.mod(head_dir + np.pi, 2 * np.pi) - np.pi
        
        traj = {
            'init_x': position[:, 1, 0],
            'init_y': position[:, 1, 1],
            'ego_v': velocity[:, 1:-1],
            'target_hd': head_dir[:, 1:-1],
            'target_x': position[:, 2:, 0],
            'target_y': position[:, 2:, 1],
        }
        
        return traj

class PlaceCells:
    """
    Place cell implementation

    Same as methods in https://github.com/ganguli-lab/grid-pattern-formation/tree/master
    """
    
    def __init__(
        self,
        num_cells: int = 512,
        box_width: float = 2.2,
        box_height: float = 2.2,
        sigma: float = 0.12,
        surround_scale: float = 2.0,
        surround_amplitude: float = 0.5, 
        periodic: bool = False,
        DoG: bool = True,
        device: str = 'cpu'
    ):
        self.num_cells = num_cells
        self.box_width = box_width
        self.box_height = box_height
        self.sigma = sigma
        self.surround_scale = surround_scale
        self.surround_amplitude = surround_amplitude
        self.periodic = periodic
        self.DoG = DoG
        self.device = device
        
        # Place cell centers uniformly distributed in box
        np.random.seed(0)
        us_x = np.random.uniform(-box_width/2, box_width/2, num_cells)
        us_y = np.random.uniform(-box_height/2, box_height/2, num_cells)
        self.us = torch.tensor(
            np.stack([us_x, us_y], axis=-1),
            dtype=torch.float32,
            device=device
        )  # (num_cells, 2)
        
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def get_activation(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Get place cell activation at a given position

        Returns activations with shape [..., num_cells]
        """
        pos_shape = pos.shape[:-1]
        pos_flat = pos.reshape(-1, 2) # (N, 2)
        
        # Compute squared distances
        d = torch.abs(pos_flat[:, None, :] - self.us[None, :, :]) # (N, num_cells, 2)
        
        if self.periodic:
            dx = d[:, :, 0]
            dy = d[:, :, 1]
            dx = torch.minimum(dx, self.box_width - dx)
            dy = torch.minimum(dy, self.box_height - dy)
            d = torch.stack([dx, dy], dim=-1)
        
        norm2 = (d ** 2).sum(-1) # (N, num_cells)
        
        outputs = self.softmax(-norm2 / (2 * self.sigma ** 2))
        
        if self.DoG:
            outputs = outputs - self.softmax(-norm2 / (2 * self.surround_scale * self.sigma ** 2))
            
            # Shift and scale outputs so that they are in [0,1]
            min_output, _ = outputs.min(-1, keepdim=True)
            outputs = outputs + torch.abs(min_output)
            outputs = outputs / outputs.sum(-1, keepdim=True)
        
        # Reshape back
        return outputs.reshape(*pos_shape, self.num_cells)
