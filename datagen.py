import numpy as np
import torch
import os
from pathlib import Path

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells


class _PlaceCellsWrapper:
    """
    Compatability for analysis code
    """
    def __init__(self, us: torch.Tensor):
        self.us = us

class GridCellDataGenerator:
    """
    Generates batch of trajectories using RatInABox

        Box size: 2.2m x 2.2m centered at origin (-1.1 to 1.1)
        dt: 0.02s (20ms timesteps)
        Place cell width: 0.12m
        DoG (Difference of Gaussians) for place cells
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
        """
        Initialize data generator
        
        Args:
            n_place_cells: Number of place cells (512)
            box_size: Size of square environment in meters (2.2)
            dt: Timestep in seconds (0.02)
            place_cell_width: Place cell width in meters (0.12)
            surround_scale: Scale factor for surround inhibition in DoG (2.0)
            surround_amplitude: Amplitude of surround inhibition (0.5)
            DoG: Use Difference of Gaussians for place cells (True)
            sequence_length: Number of timesteps per trajectory (20)
            batch_size: Default batch size (200)
            periodic: Use periodic boundary conditions (False)
            device: PyTorch device for tensors ('cpu')
        """
        self.n_place_cells = n_place_cells
        self.box_size = box_size
        self.dt = dt
        self.place_cell_width = place_cell_width
        self.surround_scale = surround_scale
        self.surround_amplitude = surround_amplitude
        self.DoG = DoG
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.periodic = periodic
        self.device = device
        
        import ratinabox
        original_verbose = ratinabox.verbose
        ratinabox.verbose = False
        
        # Create RatInABox environment
        boundary_conditions = 'periodic' if periodic else 'solid'
        self.env = Environment(params={
            'scale': box_size, # environment scale in meters
            'boundary_conditions': boundary_conditions,
        })
        
        # Agent for place cell initialization (parameters adjusted to match ganguli lab implementation)
        self.mean_speed = 1.02
        self.template_agent = Agent(self.env, params={
            'dt': dt,
            'speed_mean': self.mean_speed,
            'speed_std': 0.53,
            'speed_coherence_time': 0.5,
            'rotational_velocity_std': 120 * np.pi / 180,
            'rotational_velocity_coherence_time': 0.08,
            'thigmotaxis': 0.5,
        })
        
        np.random.seed(0)
        
        # Sample place cell centers uniformly in the environment
        place_cell_centres = self.env.sample_positions(n=n_place_cells, method='uniform_jitter')
        
        if DoG:
            pc_description = 'diff_of_gaussians'
        else:
            pc_description = 'gaussian'
        
        self.place_cells_riab = PlaceCells(self.template_agent, params={
            'n': n_place_cells,
            'widths': place_cell_width,
            'place_cell_centres': place_cell_centres,
            'description': pc_description,
            'min_fr': 0.0,
            'max_fr': 1.0,
            'wall_geometry': 'euclidean',
        })
        
        # Store place cell centers for position decoding
        # Convert from RatInABox coordinates [0, box_size] to centered [-box_size/2, box_size/2]
        self.us = torch.tensor(
            place_cell_centres - box_size / 2,
            dtype=torch.float32,
            device=device
        )
        
        # Compatability
        self.place_cells = _PlaceCellsWrapper(self.us)
        
        np.random.seed(None)
        
        ratinabox.verbose = original_verbose
        
        mean_displacement = self.mean_speed * dt
        
        print(f"RatInABox Data Generator initialized:")
        print(f"Environment: {box_size}m x {box_size}m, boundary={boundary_conditions}")
        print(f"Timestep: {dt}s ({int(1/dt)} Hz)")
        print(f"Place cells: {n_place_cells} cells, cell width={place_cell_width}m, DoG={DoG}")
        print(f"Mean speed: {self.mean_speed:.3f} m/s")
        print(f"Expected displacement/step: ~{mean_displacement:.4f}m")
        
        self._cache_file = None
        self._cached_velocities = None
        self._cached_init_pc = None
        self._cached_target_pc = None
        self._cached_target_pos = None
        self._cache_index = 0
        self._use_cache = False
        
    def pregenerate_dataset(
        self,
        n_trajectories: int = 100000,
        seq_length: int = None,
        cache_file: str = None,
        show_progress: bool = True
    ):
        """
        Pre-generate a large dataset of trajectories and save
        """
        if seq_length is None:
            seq_length = self.sequence_length
        if cache_file is None:
            cache_dir = Path(__file__).parent / "data_cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / f"trajectories_n{n_trajectories}_seq{seq_length}_pc{self.n_place_cells}.npz"
        
        print(f"Pre-generating {n_trajectories} trajectories (seq_length={seq_length})")
        
        # Pre-allocate arrays
        all_velocities = np.zeros((n_trajectories, seq_length, 2), dtype=np.float32)
        all_init_pos = np.zeros((n_trajectories, 2), dtype=np.float32)
        all_target_pos = np.zeros((n_trajectories, seq_length, 2), dtype=np.float32)
        
        import ratinabox
        ratinabox.verbose = False
        
        # Generate trajectories with progress reporting
        for i in range(n_trajectories):
            if show_progress and (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{n_trajectories} trajectories ({100*(i+1)/n_trajectories:.1f}%)")
            
            positions, velocities = self._generate_single_trajectory(seq_length)
            
            # Convert to centered coordinates
            positions_centered = positions - self.box_size / 2
            
            all_velocities[i] = velocities
            all_init_pos[i] = positions_centered[0]
            all_target_pos[i] = positions_centered[1:]
        
        print("Computing place cell activations")
        
        # Compute place cell activations in batches
        batch_size = 10000
        all_init_pc = np.zeros((n_trajectories, self.n_place_cells), dtype=np.float32)
        all_target_pc = np.zeros((n_trajectories, seq_length, self.n_place_cells), dtype=np.float32)
        
        for start in range(0, n_trajectories, batch_size):
            end = min(start + batch_size, n_trajectories)
            all_init_pc[start:end] = self._get_place_cell_activation(all_init_pos[start:end])
            all_target_pc[start:end] = self._get_place_cell_activation(all_target_pos[start:end])
            if show_progress:
                print(f"Computed place cells for {end}/{n_trajectories} trajectories")
        
        print(f"Saving to {cache_file}")
        np.savez_compressed(
            cache_file,
            velocities=all_velocities,
            init_pc=all_init_pc,
            target_pc=all_target_pc,
            target_pos=all_target_pos,
            n_place_cells=self.n_place_cells,
            box_size=self.box_size,
            dt=self.dt,
            place_cell_width=self.place_cell_width,
            seq_length=seq_length
        )
        
        file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
        print(f"Dataset saved, Size: {file_size_mb:.1f} MB")
        print(f"File: {cache_file}")
        
        return str(cache_file)
    
    def load_cache(self, cache_file: str = None, auto_find: bool = True):
        """
        Load pre-generated dataset from disk for fast training
        
        Returns true if loaded successfully
        """
        if cache_file is None and auto_find:
            cache_dir = Path(__file__).parent / "data_cache"
            if cache_dir.exists():
                pattern = f"trajectories_*_seq{self.sequence_length}_pc{self.n_place_cells}.npz"
                matches = list(cache_dir.glob(pattern))
                if matches:
                    # Use most trajectories
                    cache_file = max(matches, key=lambda p: p.stat().st_size)
                    print(f"Auto-found cache file: {cache_file}")
        if cache_file is None or not Path(cache_file).exists():
            print("No cache found")
            return False
        
        print(f"Loading cached dataset from {cache_file}")
        data = np.load(cache_file)
        
        if int(data['n_place_cells']) != self.n_place_cells:
            print(f"Warning: Cache has {data['n_place_cells']} place cells, expected {self.n_place_cells}")
            return False
        if int(data['seq_length']) != self.sequence_length:
            print(f"Warning: Cache has seq_length={data['seq_length']}, expected {self.sequence_length}")
            return False
            
        self._cached_velocities = data['velocities']
        self._cached_init_pc = data['init_pc']
        self._cached_target_pc = data['target_pc']
        self._cached_target_pos = data['target_pos']
        self._cache_index = 0
        self._use_cache = True
        self._cache_file = cache_file
        
        n_cached = len(self._cached_velocities)
        print(f"Loaded {n_cached} pre-generated trajectories")
        
        return True
        
    def _generate_single_trajectory(self, seq_length: int):
        """
        Generate a single trajectory using RatInABox
        """
        import ratinabox
        original_verbose = ratinabox.verbose
        ratinabox.verbose = False
        
        agent = Agent(self.env, params={
            'dt': self.dt,
            'speed_mean': 1.02,
            'speed_std': 0.53,
            'speed_coherence_time': 0.5,
            'rotational_velocity_std': 120 * np.pi / 180,
            'rotational_velocity_coherence_time': 0.08,
            'thigmotaxis': 0.5,
        })
        
        agent.history['pos'] = []
        agent.history['vel'] = []
        agent.history['t'] = []
        
        # Store initial position
        positions = [agent.pos.copy()]
        velocities = []
        
        # Simulate trajectory
        for _ in range(seq_length):
            agent.update()
            positions.append(agent.pos.copy())
            velocities.append(agent.velocity.copy() * self.dt)
        
        ratinabox.verbose = original_verbose
        
        return np.array(positions), np.array(velocities)
    
    def _get_place_cell_activation(self, positions: np.ndarray) -> np.ndarray:
        """
        Get place cell activations for given positions
        """
        # Convert from centered coordinates to RatInABox coordinates [0, box_size]
        positions_riab = positions + self.box_size / 2
        
        # Flatten positions for processing
        original_shape = positions.shape[:-1]
        pos_flat = positions_riab.reshape(-1, 2)
        
        # Get activations for each position
        activations = self.place_cells_riab.get_state(evaluate_at=None, pos=pos_flat)
        
        # From (n_cells, n_positions) to (n_positions, n_cells)
        activations = activations.T
        
        activations = activations.reshape(*original_shape, self.n_place_cells)
        
        return activations
        
    def generate_batch(
        self,
        batch_size: int = None,
        seq_length: int = None
    ):
        """
        Generate a batch of training data
        """
        if batch_size is None:
            batch_size = self.batch_size
        if seq_length is None:
            seq_length = self.sequence_length
        
        # Use cached data if available
        if self._use_cache:
            return self._generate_batch_from_cache(batch_size, seq_length)
        
        # Generate on spot
        return self._generate_batch_online(batch_size, seq_length)
    
    def _generate_batch_from_cache(self, batch_size: int, seq_length: int):
        """
        Sample a batch from pre-generated cached data
        """
        n_cached = len(self._cached_velocities)
        
        # Random sampling from cache
        indices = np.random.randint(0, n_cached, size=batch_size)
        
        # Extract batch
        velocities_batch = self._cached_velocities[indices] # (batch, seq, 2)
        init_pc_np = self._cached_init_pc[indices] # (batch, n_cells)
        target_pc_np = self._cached_target_pc[indices] # (batch, seq, n_cells)
        target_pos = self._cached_target_pos[indices] # (batch, seq, 2)
        
        # Convert to tensor
        velocity = torch.tensor(
            velocities_batch.transpose(1, 0, 2), # (seq, batch, 2)
            dtype=torch.float32,
            device=self.device
        )
        
        init_pc = torch.tensor(
            init_pc_np, # (batch, n_cells)
            dtype=torch.float32,
            device=self.device
        )
        
        target_pc = torch.tensor(
            target_pc_np.transpose(1, 0, 2), # (seq, batch, n_cells)
            dtype=torch.float32,
            device=self.device
        )
        
        target_pos_tensor = torch.tensor(
            target_pos.transpose(1, 0, 2), # (seq, batch, 2)
            dtype=torch.float32,
            device=self.device
        )
        
        return velocity, init_pc, target_pc, target_pos_tensor
    
    def _generate_batch_online(self, batch_size: int, seq_length: int):
        """
        Generate trajectories on spot using RatInABox
        """
        all_positions = []
        all_velocities = []
        
        for _ in range(batch_size):
            positions, velocities = self._generate_single_trajectory(seq_length)
            all_positions.append(positions)
            all_velocities.append(velocities)
        
        positions_batch = np.stack(all_positions, axis=0) # (batch, seq+1, 2)
        velocities_batch = np.stack(all_velocities, axis=0) # (batch, seq, 2)
        
        # From [0, box_size] to [-box_size/2, box_size/2]
        positions_centered = positions_batch - self.box_size / 2
        
        # Extract initial and target positions
        init_pos = positions_centered[:, 0, :] # (batch, 2)
        target_pos = positions_centered[:, 1:, :] # (batch, seq, 2)
        
        # Get place cell activations
        init_pc_np = self._get_place_cell_activation(init_pos) # (batch, n_cells)
        target_pc_np = self._get_place_cell_activation(target_pos) # (batch, seq, n_cells)
        
        # Convert to tensors

        # (seq, batch, 2)
        velocity = torch.tensor(
            velocities_batch.transpose(1, 0, 2),
            dtype=torch.float32,
            device=self.device
        )
        
        # (batch, n_cells)
        init_pc = torch.tensor(
            init_pc_np,
            dtype=torch.float32,
            device=self.device
        )
        
        # (seq, batch, n_cells)
        target_pc = torch.tensor(
            target_pc_np.transpose(1, 0, 2),
            dtype=torch.float32,
            device=self.device
        )
        
        # (seq, batch, 2)
        target_pos_tensor = torch.tensor(
            target_pos.transpose(1, 0, 2),
            dtype=torch.float32,
            device=self.device
        )
        
        return velocity, init_pc, target_pc, target_pos_tensor
    
    def compute_position_error(self, pred_logits, true_pos, k=3):
        """
        Calculate position decoding error in centimeters using top k decoding
        """
        with torch.no_grad():
            # Get indices of top-k predictions
            _, idxs = torch.topk(pred_logits, k=k, dim=-1) # (seq, batch, k)
            
            # Predicted position as mean of top k
            pred_pos = self.us[idxs].mean(dim=-2) # (seq, batch, 2)
            
            # Compute euclidean distance
            error_m = torch.sqrt(((pred_pos - true_pos) ** 2).sum(dim=-1)).mean()
            
            # Convert to centimeters
            return error_m.item() * 100

def main():
    """
    For generating trajectory datasets
    
    python datagen.py --generate --n 100000
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pre-generate trajectory datasets for fast training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Default: python datagen.py --generate --n 100000"""
    )
    
    parser.add_argument('--generate', action='store_true',
                        help='Generate and save trajectory dataset')
    parser.add_argument('--n', type=int, default=100000,
                        help='Number of trajectories to generate (default: 100000)')
    parser.add_argument('--seq', type=int, default=20,
                        help='Sequence length (default: 20)')
    parser.add_argument('--pc', type=int, default=512,
                        help='Number of place cells (default: 512)')
    parser.add_argument('--box', type=float, default=2.2,
                        help='Box size in meters (default: 2.2)')
    parser.add_argument('--dt', type=float, default=0.02,
                        help='Timestep in seconds (default: 0.02)')
    parser.add_argument('--pc-width', type=float, default=0.12,
                        help='Place cell width/sigma (default: 0.12)')
    parser.add_argument('--test', action='store_true',
                        help='Quick test of data generator')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark online vs cached generation speed')
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing GridCellDataGenerator")
        gen = GridCellDataGenerator(
            n_place_cells=args.pc,
            box_size=args.box,
            dt=args.dt,
            place_cell_width=args.pc_width,
            sequence_length=args.seq
        )
        
        print("\nGenerating small batch (10 trajectories)")
        import time
        start = time.time()
        v, init_pc, target_pc, target_pos = gen.generate_batch(batch_size=10)
        elapsed = time.time() - start
        
        print(f"Time: {elapsed:.3f}s")
        print(f"velocity shape: {v.shape}")
        print(f"init_pc shape: {init_pc.shape}")
        print(f"target_pc shape: {target_pc.shape}")
        print(f"target_pos shape: {target_pos.shape}")
        return
    
    if args.benchmark:
        print("Benchmarking generation speed")
        import time
        
        gen = GridCellDataGenerator(
            n_place_cells=args.pc,
            box_size=args.box,
            dt=args.dt,
            place_cell_width=args.pc_width,
            sequence_length=args.seq
        )
        
        # Test online generation
        print("\nOnline generation (RatInABox during run):")
        start = time.time()
        for _ in range(5):
            gen.generate_batch(batch_size=200)
        online_time = (time.time() - start) / 5
        print(f"Average time per batch (200 trajectories): {online_time:.3f}s")
        
        # Check if cache exists
        if gen.load_cache():
            print("\nCached generation (pre-generated data)")
            start = time.time()
            for _ in range(100):
                gen.generate_batch(batch_size=200)
            cached_time = (time.time() - start) / 100
            print(f"Average time per batch (200 trajectories): {cached_time:.5f}s")
        else:
            print("\nNo cache found. Run with --generate first to benchmark cached speed")
        return
    
    if args.generate:
        gen = GridCellDataGenerator(
            n_place_cells=args.pc,
            box_size=args.box,
            dt=args.dt,
            place_cell_width=args.pc_width,
            sequence_length=args.seq
        )
        
        gen.pregenerate_dataset(
            n_trajectories=args.n,
            seq_length=args.seq,
            show_progress=True
        )
        
        print("\n" + "="*60)
        print("To use in training, the cache will be loaded automatically.")
        print("Or manually call: data_gen.load_cache()")
        print("="*60)
        return
    
    # If no arguments, print help
    parser.print_help()

if __name__ == '__main__':
    main()
