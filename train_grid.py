import argparse
import os
import random
import time 
from datetime import datetime

import torch
import numpy as np
import wandb

from grid_network import GridNetwork, get_device
from datagen import GridCellDataGenerator
from analyze import (
    compute_grid_scores_from_model,
    create_top_ratemaps_figure,
    create_grid_score_histogram_figure,
    create_topographic_orientation_map_figure,
    create_topographic_phase_map_figure,
    create_topographic_scale_map_figure,
    create_trajectory_decoding_figure
)

# W&B configuration
WANDB_ENTITY = 'topogrid'
WANDB_PROJECT = 'TopographicGridCells'


def save_checkpoint(model, optimizer, step, loss, save_dir, log_to_wandb=True):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'model_step_{step}.pth')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f'Saved checkpoint: {path}')
    
    if log_to_wandb and wandb.run is not None:
        artifact = wandb.Artifact(
            name=f'model-checkpoint-step-{step}',
            type='model',
            description=f'Model checkpoint at step {step}',
            metadata={'step': step, 'loss': loss}
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
    
    return path

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step'], checkpoint['loss']

def set_seed(seed):
    """Set seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return seed

def log_validation_metrics(model, data_gen, step, box_size=2.2):
    """
    Logging validation metrics to wandb
    
    Logs:
        Number of grid cells above threshold 0.3
        Top 10 ratemaps by score
        Grid score mean
        Grid score histogram
        Topographic phase map
        Topographic scale map
        Topographic orientation map
        Trajectory decoding visualization and error
    """
    print(f'\nComputing validation metrics at step {step}')
    
    # Compute grid scores and ratemaps
    grid_scores, ratemaps = compute_grid_scores_from_model(
        model, data_gen, n_batches=50, batch_size=200, seq_length=20, res=32
    )
    
    # Metrics
    n_above_threshold = int(np.sum(grid_scores > 0.3))
    grid_score_mean = float(grid_scores.mean())
    grid_score_max = float(grid_scores.max())
    grid_score_median = float(np.median(grid_scores))
    
    # Create figures
    top_ratemaps_fig = create_top_ratemaps_figure(ratemaps, grid_scores, n_top=10, box_size=box_size)
    histogram_fig = create_grid_score_histogram_figure(grid_scores, threshold=0.3)
    phase_map_fig = create_topographic_phase_map_figure(ratemaps, grid_scores, mask_threshold=0.0)
    trajectory_fig, trajectory_error_cm = create_trajectory_decoding_figure(
        model, data_gen, n_trajectories=25, seq_length=20
    )
    scale_map_fig = create_topographic_scale_map_figure(ratemaps, grid_scores, mask_threshold=0.0)
    orientation_map_fig = create_topographic_orientation_map_figure(ratemaps, grid_scores, mask_threshold=0.0)
    
    wandb.log({
        'validation/n_grid_cells_above_0.3': n_above_threshold,
        'validation/grid_score_mean': grid_score_mean,
        'validation/grid_score_max': grid_score_max,
        'validation/grid_score_median': grid_score_median,
        'validation/trajectory_error_cm': trajectory_error_cm,
        'validation/top_10_ratemaps': wandb.Image(top_ratemaps_fig),
        'validation/grid_score_histogram': wandb.Image(histogram_fig),
        'validation/topographic_phase_map': wandb.Image(phase_map_fig),
        'validation/topographic_scale_map': wandb.Image(scale_map_fig),
        'validation/topographic_orientation_map': wandb.Image(orientation_map_fig),
        'validation/trajectory_decoding': wandb.Image(trajectory_fig),
    }, step=step)
    
    import matplotlib.pyplot as plt
    plt.close(top_ratemaps_fig)
    plt.close(histogram_fig)
    plt.close(phase_map_fig)
    plt.close(trajectory_fig)
    plt.close(scale_map_fig)
    plt.close(orientation_map_fig)
    
    print(f'Grid cells above 0.3: {n_above_threshold} ({100*n_above_threshold/len(grid_scores):.1f}%)')
    print(f'Grid score mean: {grid_score_mean:.3f}, max: {grid_score_max:.3f}')
    print(f'Trajectory error: {trajectory_error_cm:.2f} cm\n')

def train(args):
    # Setup device
    device = get_device()
    
    # Set random seed for reproducibility
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    set_seed(seed)
    print(f'Random seed: {seed}')
    
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    
    # Determine training mode
    if args.dtl:
        mode_name = 'dtl'
        run_group = 'decomposed-topographic'
    elif args.topoloss:
        mode_name = 'topo'
        run_group = 'topographic'
    else:
        mode_name = 'nontopo'
        run_group = 'non-topographic'
    
    # Initialize wandb
    run_name = args.wandb_name or f"{mode_name}_{datetime.now().strftime('%m_%d_%y_%H%M')}"
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        group=run_group,
        config={
            # Training params
            'steps': args.steps,
            'batch_size': args.batch_size,
            'seq_length': args.seq_length,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'topoloss': args.topoloss,
            'dtl': args.dtl,
            'scale_weight': args.scale_weight if args.dtl else None,
            'phase_weight': args.phase_weight if args.dtl else None,
            # Model params
            'n_place_cells': args.n_place_cells,
            'n_grid_cells': args.n_grid_cells,
            'activation': args.activation,
            # Data generation params
            'box_size': args.box_size,
            'dt': args.dt,
            'pc_width': args.pc_width,
            'surround_scale': args.surround_scale,
            'surround_amplitude': args.surround_amplitude,
            'DoG': args.DoG,
            # System info
            'device': str(device),
            'seed': seed,
        },
        tags=[run_group],
        resume='allow' if args.resume else None,
    )
    
    print(f'\nGrid Cell Training')
    print(f'W&B Run: {wandb.run.name} ({wandb.run.url})')
    print(f'Device: {device}')
    print(f'Steps: {args.steps:,}')
    print(f'Batch size: {args.batch_size}')
    print(f'Sequence length: {args.seq_length}')
    print(f'Place cells: {args.n_place_cells}')
    print(f'Grid cells: {args.n_grid_cells}')
    print(f'Learning rate: {args.lr}')
    print(f'Weight decay: {args.weight_decay}')
    print()
    
    data_gen = GridCellDataGenerator(
        n_place_cells=args.n_place_cells,
        box_size=args.box_size,
        dt=args.dt,
        place_cell_width=args.pc_width,
        surround_scale=args.surround_scale,
        surround_amplitude=args.surround_amplitude,
        DoG=args.DoG,
        device=device
    )
    
    model = GridNetwork(
        Np=args.n_place_cells,
        Ng=args.n_grid_cells,
        weight_decay=args.weight_decay,
        activation=args.activation
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')
    
    wandb.config.update({'n_params': n_params})
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    wandb.watch(model, log='all', log_freq=args.print_every)
    
    # Resume if requested
    start_step = 0
    if args.resume:
        ckpt_path = args.resume
        if ckpt_path and os.path.exists(ckpt_path):
            start_step, _ = load_checkpoint(model, optimizer, ckpt_path)
            print(f'Resumed from step {start_step:,} ({ckpt_path})')
        else:
            print('No checkpoint found, continuing from scratch')
    
    # Training loop
    model.train()
    losses = []
    start_time = time.time()
    
    print(f'\nStarting training from step {start_step:,}\n')
    
    for step in range(start_step, args.steps):
        velocity, init_pc, target_pc, positions = data_gen.generate_batch(
            batch_size=args.batch_size,
            seq_length=args.seq_length
        )
        
        optimizer.zero_grad()
        
        # Loss based on args
        if args.dtl:
            loss, metrics = model.compute_dtl_loss(
                velocity, init_pc, target_pc,
                scale_weight=args.scale_weight,
                phase_weight=args.phase_weight
            )
        elif args.topoloss:
            loss, metrics = model.compute_topographic_loss(velocity, init_pc, target_pc)
        else:
            loss, metrics = model.compute_loss(velocity, init_pc, target_pc)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(metrics['total_loss'])
        
        # Logging
        if (step + 1) % args.print_every == 0:
            avg_loss = np.mean(losses[-args.print_every:])
            elapsed = time.time() - start_time
            steps_per_sec = (step - start_step + 1) / elapsed
            
            with torch.no_grad():
                logits, _ = model(velocity, init_pc)
                error_cm = data_gen.compute_position_error(logits, positions)
            
            # Metrics to wandb
            log_dict = {
                'train/total_loss': avg_loss,
                'train/ce_loss': metrics['ce_loss'],
                'train/reg_loss': metrics['reg_loss'],
                'train/position_error_cm': error_cm,
                'train/steps_per_sec': steps_per_sec,
            }
            if args.topoloss:
                log_dict['train/topo_loss'] = metrics.get('topo_loss', 0.0)
            if args.dtl:
                log_dict['train/dtl_loss'] = metrics.get('dtl_loss', 0.0)
                log_dict['train/scale_loss'] = metrics.get('scale_loss', 0.0)
                log_dict['train/phase_loss'] = metrics.get('phase_loss', 0.0)
            wandb.log(log_dict, step=step + 1)
            
            print(f'Step {step+1:>6,} | Loss: {avg_loss:.4f} | '
                  f'CE: {metrics["ce_loss"]:.4f} | Reg: {metrics["reg_loss"]:.4f} | ' +
                  (f'Topo: {metrics.get("topo_loss", 0.0):.4f} | ' if args.topoloss else '') +
                  (f'DTL: {metrics.get("dtl_loss", 0.0):.4f} (S:{metrics.get("scale_loss", 0.0):.4f} P:{metrics.get("phase_loss", 0.0):.4f}) | ' if args.dtl else '') +
                  f'Error: {error_cm:.1f}cm | {steps_per_sec:.1f} steps/s')
        
        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, step + 1, losses[-1], args.save_dir)
        
        # Validation logging every 10,000 steps
        if (step + 1) % 10000 == 0:
            log_validation_metrics(model, data_gen, step + 1, box_size=args.box_size)
    
    # Final save
    save_checkpoint(model, optimizer, args.steps, losses[-1], args.save_dir)
    
    total_time = time.time() - start_time
    final_loss = np.mean(losses[-100:])
    
    wandb.run.summary['final_loss'] = final_loss
    wandb.run.summary['total_time_minutes'] = total_time / 60
    wandb.run.summary['total_steps'] = args.steps
    
    final_artifact = wandb.Artifact(
        name='model-final',
        type='model',
        description='Final trained model',
        metadata={
            'final_loss': final_loss,
            'total_steps': args.steps,
            'topoloss': args.topoloss
        }
    )
    final_model_path = os.path.join(args.save_dir, f'model_step_{args.steps}.pth')
    final_artifact.add_file(final_model_path)
    wandb.log_artifact(final_artifact)
    
    wandb.finish()
    
    print(f'\nTraining Complete')
    print(f'Total time: {total_time/60:.1f} minutes')
    print(f'Final loss: {final_loss:.4f}')
    print(f'Checkpoints saved to: {args.save_dir}')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train grid cell network')

    parser.add_argument('--topoloss', action='store_true', default=False,
                        help='Use topographic loss (default: False)')
    parser.add_argument('--dtl', action='store_true', default=False,
                        help='Use decomposed topographic loss (default: False)')
    parser.add_argument('--scale_weight', type=float, default=1.0,
                        help='Weight for scale topography loss in DTL (default: 1.0)')
    parser.add_argument('--phase_weight', type=float, default=0.1,
                        help='Weight for phase diversity loss in DTL (default: 0.1)')
    
    parser.add_argument('--steps', type=int, default=50000,
                        help='Number of training steps (default: 50000)')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size (default: 200)')
    parser.add_argument('--seq_length', type=int, default=20,
                        help='Sequence length (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    
    parser.add_argument('--n_place_cells', type=int, default=512,
                        help='Number of place cells (default: 512)')
    parser.add_argument('--n_grid_cells', type=int, default=4096,
                        help='Number of grid cells/hidden units (default: 4096)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay on recurrent weights (default: 1e-4)')
    parser.add_argument('--activation', type=str, default='relu',
                        help='RNN activation function (default: relu)')
    
    parser.add_argument('--box_size', type=float, default=2.2,
                        help='Environment size in meters (default: 2.2)')
    parser.add_argument('--dt', type=float, default=0.02,
                        help='Timestep in seconds (default: 0.02)')
    parser.add_argument('--pc_width', type=float, default=0.12,
                        help='Place cell width/sigma (default: 0.12)')
    parser.add_argument('--surround_scale', type=float, default=2.0,
                        help='DoG surround scale (default: 2.0)')
    parser.add_argument('--surround_amplitude', type=float, default=0.5,
                        help='DoG surround inhibition strength (default: 0.5)')
    parser.add_argument('--DoG', action='store_true', default=True,
                        help='Use Difference of Gaussians (default: True)')
    parser.add_argument('--no_DoG', action='store_false', dest='DoG',
                        help='Disable Difference of Gaussians')
    
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory for checkpoints (default: checkpoints)')
    parser.add_argument('--save_every', type=int, default=5000,
                        help='Save checkpoint every N steps (default: 5000)')
    parser.add_argument('--print_every', type=int, default=500,
                        help='Print progress every N steps (default: 500)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (path)')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Custom name for wandb run (optional)')
    parser.add_argument('--wandb_offline', action='store_true', default=False,
                        help='Run wandb in offline mode')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: random)')
    
    args = parser.parse_args()
    
    # Create save directory
    if not args.resume:
        if args.dtl:
            mode = 'dtl'
        elif args.topoloss:
            mode = 'topo'
        else:
            mode = 'nontopo'
        timestamp = f"{mode}_{datetime.now().strftime('%m_%d_%y_%H%M')}"
        args.save_dir = os.path.join(args.save_dir, timestamp)
    
    train(args)

if __name__ == '__main__':
    main()
