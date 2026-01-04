import argparse
import os
import time 
from datetime import datetime

import torch
import numpy as np

from grid_network import GridNetwork, get_device
from datagen import GridCellDataGenerator


def save_checkpoint(model, optimizer, step, loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'model_step_{step}.pth')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f'Saved checkpoint: {path}')
    return path

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step'], checkpoint['loss']

def train(args):
    # Setup device
    device = get_device()
    print(f'\nGrid Cell Training')
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
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
        loss, metrics = (model.compute_loss(velocity, init_pc, target_pc) if not args.topoloss 
                         else model.compute_topographic_loss(velocity, init_pc, target_pc))
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
            
            print(f'Step {step+1:>6,} | Loss: {avg_loss:.4f} | '
                  f'CE: {metrics["ce_loss"]:.4f} | Reg: {metrics["reg_loss"]:.4f} | ' +
                  (f'Topo: {metrics.get("topo_loss", 0.0):.4f} | ' if args.topoloss else '') +
                  f'Error: {error_cm:.1f}cm | {steps_per_sec:.1f} steps/s')
        
        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, step + 1, losses[-1], args.save_dir)
    
    # Final save
    save_checkpoint(model, optimizer, args.steps, losses[-1], args.save_dir)
    
    total_time = time.time() - start_time
    print(f'\nTraining Complete')
    print(f'Total time: {total_time/60:.1f} minutes')
    print(f'Final loss: {np.mean(losses[-100:]):.4f}')
    print(f'Checkpoints saved to: {args.save_dir}')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train grid cell network')

    parser.add_argument('--topoloss', action='store_true', default=False,
                        help='Use topographic loss (default: False)')
    
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
    
    args = parser.parse_args()
    
    # Create save directory
    if not args.resume:
        timestamp = datetime.now().strftime('%m_%d_%H%M')
        args.save_dir = os.path.join(args.save_dir, timestamp)
    
    train(args)

if __name__ == '__main__':
    main()
