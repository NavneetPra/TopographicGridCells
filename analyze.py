import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import scipy.signal
import scipy.ndimage
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.fft import fft2, fftshift

from datagen import GridCellDataGenerator
from grid_network import GridNetwork, get_device


def get_ratemap(pos_history, activations, res=32, box_width=2.2, box_height=2.2):
    """
    Creates spatial ratemap for sinngle grid cell
    
        pos_history: (N, 2) positions relative to center (ex range of X is -box_width/2 to box_width/2)
        activations: (N,) activation values for each position
    
    Returns (res, res) ratemap
    """
    # Edge coordinates
    edges_x = np.linspace(-box_width/2, box_width/2, res + 1)
    edges_y = np.linspace(-box_height/2, box_height/2, res + 1)

    occupancy, _, _ = np.histogram2d(pos_history[:,0], pos_history[:,1], bins=[edges_x, edges_y])
    activity, _, _ = np.histogram2d(pos_history[:,0], pos_history[:,1], bins=[edges_x, edges_y], weights=activations)

    # Avoid division by zero
    ratemap = np.divide(activity, occupancy, out=np.zeros_like(activity), where=occupancy > 0)

    return ratemap.T

def visualize_cells(
        model_path, 
        output_folder, 
        save_imgs=True, 
        n_batches=50, 
        batch_size=200, 
        seq_length=20, 
        model_Np=512, 
        model_Ng=4096, 
        model_weight_decay=1e-4, 
        activation='relu', 
        gen_box_size=2.2, 
        gen_dt=0.02, 
        gen_pc_width=0.12, 
        gen_surround_scale=2.0, 
        gen_surround_amplitude=0.5, 
        gen_dog=True,
        res=32
    ):
    """
    Visualize and save ratemap of each grid cell of a trained model.
    """
    os.makedirs(output_folder, exist_ok=True)

    device = get_device()
    model = GridNetwork(
        Np=model_Np,
        Ng=model_Ng,
        weight_decay=model_weight_decay,
        activation=activation
    ).to(device)

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data_gen = GridCellDataGenerator(
        n_place_cells=model_Np,
        box_size=gen_box_size,
        dt=gen_dt,
        place_cell_width=gen_pc_width,
        surround_scale=gen_surround_scale,
        surround_amplitude=gen_surround_amplitude,
        DoG=gen_dog,
        device=device
    )

    # Collect activations through short batches
    print(f'Collecting grid cell activations from {n_batches} batches')
    all_g = []
    all_pos = []
    
    for i in range(n_batches):
        velocity, init_pc, _, positions = data_gen.generate_batch(
            batch_size=batch_size,
            seq_length=seq_length
        )
        
        with torch.no_grad():
            g = model.g(velocity, init_pc)
        
        # g: (seq, batch, Ng), positions: (seq, batch, 2)
        g_flat = g.reshape(-1, model.Ng).cpu().numpy()
        pos_flat = positions.reshape(-1, 2).cpu().numpy()
        
        all_g.append(g_flat)
        all_pos.append(pos_flat)
        
        if (i + 1) % 10 == 0:
            print(f'Collected {i + 1}/{n_batches} batches')
    
    g = np.concatenate(all_g, axis=0)
    positions = np.concatenate(all_pos, axis=0) 
    num_cells = g.shape[1]

    print(f'Position range: x=[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], y=[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]')

    box_width = gen_box_size
    box_height = gen_box_size
    
    print('Making ratemaps for all cells')
    ratemaps = np.zeros((num_cells, res, res))
    grid_scores = np.zeros(num_cells)
    
    for cell_idx in range(num_cells):
        activation_trace = g[:, cell_idx]
        ratemap = get_ratemap(positions, activation_trace, res=res, box_width=box_width, box_height=box_height)
        ratemaps[cell_idx] = ratemap
        grid_scores[cell_idx], _ = get_grid_score(ratemap)
        
        if (cell_idx + 1) % 500 == 0:
            print(f'Processed {cell_idx + 1}/{num_cells} cells')
    
    print(f'Grid scores range: [{grid_scores.min():.3f}, {grid_scores.max():.3f}]')
    print(f'Mean: {grid_scores.mean():.3f}, Median: {np.median(grid_scores):.3f}')
    
    # Sort by grid score
    sorted_idx = np.argsort(grid_scores)[::-1]

    if save_imgs:
        for i, cell_idx in enumerate(sorted_idx):
            plt.figure(figsize=(4,4))

            ratemap = ratemaps[cell_idx]

            plt.imshow(ratemap, origin='lower', extent=[-box_width/2, box_width/2, -box_height/2, box_height/2], cmap='jet')
            plt.title(f'Grid Cell {cell_idx} (Score: {grid_scores[cell_idx]:.3f})')
            plt.axis('off')
            plt.colorbar(label='Firing Rate')

            filename = os.path.join(output_folder, f'grid_cell_rank{i:04d}_idx{cell_idx}.png')
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

            if (i + 1) % 100 == 0:
                print(f'Saved {i + 1}/{num_cells} ratemaps')
    
    num_plots = 16
    cols = 4
    rows = 4

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle('Top 16 grid cells by score', fontsize=14)

    for i in range(num_plots):
        ax = axes[i // cols, i % cols]

        cell_idx = sorted_idx[i]
        ratemap = ratemaps[cell_idx]

        im = ax.imshow(ratemap, origin='lower', extent=[-box_width/2, box_width/2, -box_height/2, box_height/2], cmap='jet')
        ax.set_title(f'Cell {cell_idx}\nScore: {grid_scores[cell_idx]:.3f}', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    print('Visualization complete')
    
    return grid_scores, ratemaps, sorted_idx

def visualize_pred(
        model_path, 
        model_Np=512, 
        model_Ng=4096, 
        model_weight_decay=1e-4, 
        activation='relu', 
        gen_box_size=2.2, 
        gen_dt=0.02, 
        gen_pc_width=0.12, 
        gen_surround_scale=2.0, 
        gen_surround_amplitude=0.5, 
        gen_dog=True
    ):
    device = get_device()
    model = GridNetwork(
        Np=model_Np,
        Ng=model_Ng,
        weight_decay=model_weight_decay,
        activation=activation
    ).to(device)

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data_gen = GridCellDataGenerator(
        n_place_cells=model_Np,
        box_size=gen_box_size,
        dt=gen_dt,
        place_cell_width=gen_pc_width,
        surround_scale=gen_surround_scale,
        surround_amplitude=gen_surround_amplitude,
        DoG=gen_dog,
        device=device
    )

    seq_length = round((10) / 0.02) # 10 second sequence
    velocity, init_pc, _, positions = data_gen.generate_batch(
        batch_size=1,
        seq_length=seq_length
    )

    with torch.no_grad():
        place_logits, _ = model(velocity, init_pc)
        
        # Decode predicted position
        place_cell_centers = data_gen.place_cells.us 
        pred_pos = model.decode_position(place_logits, place_cell_centers)

    positions_np = positions.squeeze(1).cpu().numpy() 
    pred_pos_np = pred_pos.squeeze(1).cpu().numpy() 

    pos_error = np.sqrt(((positions_np - pred_pos_np) ** 2).sum(axis=1))
    mean_error_cm = pos_error.mean() * 100
    print(f'Mean position error: {mean_error_cm:.2f} cm')

    fig = plt.figure(figsize=(16, 12))
    
    # True vs. decoded trajectory
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(positions_np[:, 0], positions_np[:, 1], 'b-', alpha=0.7, label='True trajectory', linewidth=1)
    ax1.plot(pred_pos_np[:, 0], pred_pos_np[:, 1], 'r--', alpha=0.7, label='Decoded trajectory', linewidth=1)
    ax1.scatter(positions_np[0, 0], positions_np[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
    ax1.scatter(positions_np[-1, 0], positions_np[-1, 1], c='red', s=100, marker='x', zorder=5, label='End')
    ax1.scatter(pred_pos_np[0, 0], pred_pos_np[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
    ax1.scatter(pred_pos_np[-1, 0], pred_pos_np[-1, 1], c='red', s=100, marker='x', zorder=5, label='End')
    ax1.set_xlim(-gen_box_size/2, gen_box_size/2)
    ax1.set_ylim(-gen_box_size/2, gen_box_size/2)
    ax1.set_xlabel('X position (m)')
    ax1.set_ylabel('Y position (m)')
    ax1.set_title(f'Trajectory: True vs decoded (mean error: {mean_error_cm:.2f} cm)')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Position error over time
    ax2 = fig.add_subplot(2, 2, 2)
    time = np.arange(len(pos_error)) * gen_dt  # Convert to seconds
    ax2.plot(time, pos_error * 100, 'b-', linewidth=0.5)
    ax2.axhline(y=mean_error_cm, color='r', linestyle='--', label=f'Mean: {mean_error_cm:.2f} cm')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position error (cm)')
    ax2.set_title('Position decoding error over time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    print('Prediction visualization complete')

def visualize_trajectory_decoding(
        model_path, 
        n_trajectories=5, 
        seq_length=20, 
        model_Np=512, 
        model_Ng=4096, 
        model_weight_decay=1e-4, 
        activation='relu', 
        gen_box_size=2.2, 
        gen_dt=0.02, 
        gen_pc_width=0.12, 
        gen_surround_scale=2.0, 
        gen_surround_amplitude=0.5, 
        gen_dog=True
    ):
    """
    Visualize true vs decoded trajectories
    """
    device = get_device()
    model = GridNetwork(
        Np=model_Np,
        Ng=model_Ng,
        weight_decay=model_weight_decay,
        activation=activation
    ).to(device)

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data_gen = GridCellDataGenerator(
        n_place_cells=model_Np,
        box_size=gen_box_size,
        dt=gen_dt,
        place_cell_width=gen_pc_width,
        surround_scale=gen_surround_scale,
        surround_amplitude=gen_surround_amplitude,
        DoG=gen_dog,
        device=device
    )

    velocity, init_pc, target_pc, positions = data_gen.generate_batch(
        batch_size=n_trajectories,
        seq_length=seq_length
    )

    with torch.no_grad():
        place_logits, _ = model(velocity, init_pc)

        # Decode predicted position
        place_cell_centers = data_gen.place_cells.us  
        pred_pos = model.decode_position(place_logits, place_cell_centers)

    pos = positions.cpu().numpy()
    pred_pos = pred_pos.cpu().numpy()
    us = data_gen.place_cells.us.cpu().numpy() 

    box_width = gen_box_size
    box_height = gen_box_size

    # Create figure
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    # Plot place cell centers in background
    plt.scatter(us[:, 0], us[:, 1], s=20, alpha=0.5, c='lightgrey')

    for i in range(n_trajectories):
        plt.plot(pos[:, i, 0], pos[:, i, 1], c='black', 
                 label='True position' if i == 0 else None, linewidth=2)
        plt.plot(pred_pos[:, i, 0], pred_pos[:, i, 1], '.-', c='C1',
                 label='Decoded position' if i == 0 else None)

    plt.legend()

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-box_width / 2, box_width / 2])
    plt.ylim([-box_height / 2, box_height / 2])

    plt.tight_layout()
    plt.show()
    
    pos_error = np.sqrt(((pos - pred_pos) ** 2).sum(axis=-1))
    mean_error_cm = pos_error.mean() * 100
    print(f'Mean position error: {mean_error_cm:.2f} cm')

def calculate_sac(ratemap):
    """
    Calculate spatial autocorrelogram for ratemap
    
    Returns sac of size (2*res-1, 2*res-1)
    """
    seq1 = np.nan_to_num(ratemap)
    seq2 = seq1.copy()
    
    ones_seq1 = np.ones(seq1.shape)
    ones_seq2 = np.ones(seq2.shape)
    
    seq1_sq = np.square(seq1)
    seq2_sq = np.square(seq2)
    
    def filter2(b, x):
        stencil = np.rot90(b, 2)
        return scipy.signal.convolve2d(x, stencil, mode='full')
    
    seq1_x_seq2 = filter2(seq1, seq2)
    sum_seq1 = filter2(seq1, ones_seq2)
    sum_seq2 = filter2(ones_seq1, seq2)
    sum_seq1_sq = filter2(seq1_sq, ones_seq2)
    sum_seq2_sq = filter2(ones_seq1, seq2_sq)
    n_bins = filter2(ones_seq1, ones_seq2)
    n_bins_sq = np.square(n_bins)
    
    std_seq1 = np.power(
        np.maximum(
            np.subtract(
                np.divide(sum_seq1_sq, n_bins),
                np.divide(np.square(sum_seq1), n_bins_sq)
            ), 0
        ), 0.5
    )
    std_seq2 = np.power(
        np.maximum(
            np.subtract(
                np.divide(sum_seq2_sq, n_bins),
                np.divide(np.square(sum_seq2), n_bins_sq)
            ), 0
        ), 0.5
    )
    
    covar = np.subtract(
        np.divide(seq1_x_seq2, n_bins),
        np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq)
    )
    
    with np.errstate(divide='ignore', invalid='ignore'):
        x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
    
    x_coef = np.nan_to_num(x_coef)
    return x_coef

def get_grid_score(ratemap):
    """
    Get grid scores for a single ratemap through rotational symmetry

    Returns grid_score_60 and sac (corresponding spatial autocorrelogram)
    """
    sac = calculate_sac(ratemap)
    
    # Angles to check for symmetry
    corr_angles = [30, 45, 60, 90, 120, 135, 150]
    
    # Create ring masks for different radii
    nbins = ratemap.shape[0]
    sac_size = nbins * 2 - 1
    center = nbins - 1
    
    # Create masks with different inner and outer radii
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    
    best_score = -np.inf
    
    for mask_min, mask_max in zip(starts, ends):
        # Create ring mask
        y, x = np.ogrid[:sac_size, :sac_size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        mask = ((dist >= mask_min * nbins) & (dist <= mask_max * nbins)).astype(float)
        
        if mask.sum() == 0:
            continue
            
        # Correlations at different angles
        masked_sac = sac * mask
        ring_area = np.sum(mask)
        masked_sac_mean = np.sum(masked_sac) / ring_area
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.sum(masked_sac_centered**2) / ring_area + 1e-5
        
        corrs = {}
        for angle in corr_angles:
            rotated_sac = scipy.ndimage.rotate(sac, angle, reshape=False)
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
            corrs[angle] = cross_prod / variance
        
        # Grid score = mean of 60° and 120° minus mean of 30°, 90°, 150° (since hexagonal symmetry)
        score_60 = (corrs[60] + corrs[120]) / 2 - (corrs[30] + corrs[90] + corrs[150]) / 3
        
        if score_60 > best_score:
            best_score = score_60
    
    return best_score, sac

def compute_grid_scores(
        model_path, 
        n_cells=None, 
        res=32, 
        n_batches=50, 
        batch_size=200, 
        seq_length=20, 
        model_Np=512, 
        model_Ng=4096, 
        model_weight_decay=1e-4, 
        activation='relu', 
        gen_box_size=2.2, 
        gen_dt=0.02, 
        gen_pc_width=0.12, 
        gen_surround_scale=2.0, 
        gen_surround_amplitude=0.5, 
        gen_dog=True
    ):
    """
    Get grid scores for all cells in a model through multiple short trajectories
    
    Returns scores (array of grid scores for each cell) and ratemaps (array of ratemaps with size (n_cells, res, res))
    """
    device = get_device()
    model = GridNetwork(
        Np=model_Np,
        Ng=model_Ng,
        weight_decay=model_weight_decay,
        activation=activation
    ).to(device)

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data_gen = GridCellDataGenerator(
        n_place_cells=model_Np,
        box_size=gen_box_size,
        dt=gen_dt,
        place_cell_width=gen_pc_width,
        surround_scale=gen_surround_scale,
        surround_amplitude=gen_surround_amplitude,
        DoG=gen_dog,
        device=device
    )

    # Activations from short trajectories
    print(f'Collecting grid cell activations from {n_batches} batches')
    all_g = []
    all_pos = []
    
    for i in range(n_batches):
        velocity, init_pc, _, positions = data_gen.generate_batch(
            batch_size=batch_size,
            seq_length=seq_length
        )
        
        with torch.no_grad():
            g = model.g(velocity, init_pc)
        
        g_flat = g.reshape(-1, model.Ng).cpu().numpy()
        pos_flat = positions.reshape(-1, 2).cpu().numpy()
        
        all_g.append(g_flat)
        all_pos.append(pos_flat)
        
        if (i + 1) % 10 == 0:
            print(f'Collected {i + 1}/{n_batches} batches')
    
    all_g = np.concatenate(all_g, axis=0)
    all_pos = np.concatenate(all_pos, axis=0)
    
    print(f'Total samples: {len(all_g)}')
    
    num_cells = all_g.shape[1] if n_cells is None else min(n_cells, all_g.shape[1])
    
    print(f'Computing grid scores for {num_cells} cells')
    
    scores = np.zeros(num_cells)
    ratemaps = np.zeros((num_cells, res, res))
    
    for i in range(num_cells):
        ratemap = get_ratemap(all_pos, all_g[:, i], res=res)
        ratemaps[i] = ratemap
        scores[i], _ = get_grid_score(ratemap)
        
        if (i + 1) % 500 == 0:
            print(f'Processed {i + 1}/{num_cells} cells')
    
    print(f'Grid scores computed. Range: [{scores.min():.3f}, {scores.max():.3f}]')
    print(f'Mean: {scores.mean():.3f}, Median: {np.median(scores):.3f}')
    
    return scores, ratemaps

def visualize_grid_scores(model_path, n_examples=8, n_cells=None, res=32, gen_box_size=2.2):
    """
    Get all grid scores and visualize the cells with the highest and lowest scores.

    Returns scores and ratemaps
    """
    scores, ratemaps = compute_grid_scores(model_path, n_cells=n_cells, res=res)
    
    # Sort cells by grid score
    sorted_idx = np.argsort(scores)[::-1]
    high_idx = sorted_idx[:n_examples]
    low_idx = sorted_idx[-n_examples:][::-1]
    
    box_width = gen_box_size
    box_height = gen_box_size
    
    fig, axes = plt.subplots(4, n_examples, figsize=(2.5 * n_examples, 10))
    fig.suptitle('Grid score analysis: top vs bottom cells', fontsize=14)
    
    # High grid score cells ratemaps
    for i, idx in enumerate(high_idx):
        ax = axes[0, i]
        im = ax.imshow(ratemaps[idx], origin='lower', 
                       extent=[-box_width/2, box_width/2, -box_height/2, box_height/2],
                       cmap='jet')
        ax.set_title(f'Cell {idx}\nScore: {scores[idx]:.3f}', fontsize=9)
        ax.axis('off')
    axes[0, 0].set_ylabel('High score\nratemaps', fontsize=10)
    
    # High grid score cells autocorrelograms
    for i, idx in enumerate(high_idx):
        ax = axes[1, i]
        _, sac = get_grid_score(ratemaps[idx])
        ax.imshow(sac, origin='lower', cmap='jet')
        ax.axis('off')
    axes[1, 0].set_ylabel('High score\nautocorr', fontsize=10)
    
    # Low grid score cells ratemaps
    for i, idx in enumerate(low_idx):
        ax = axes[2, i]
        im = ax.imshow(ratemaps[idx], origin='lower',
                       extent=[-box_width/2, box_width/2, -box_height/2, box_height/2],
                       cmap='jet')
        ax.set_title(f'Cell {idx}\nScore: {scores[idx]:.3f}', fontsize=9)
        ax.axis('off')
    axes[2, 0].set_ylabel('Low score\nratemaps', fontsize=10)
    
    # Low grid score cells autocorrelograms
    for i, idx in enumerate(low_idx):
        ax = axes[3, i]
        _, sac = get_grid_score(ratemaps[idx])
        ax.imshow(sac, origin='lower', cmap='jet')
        ax.axis('off')
    axes[3, 0].set_ylabel('Low score\nautocorr', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Histogram of grid scores
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0.3, color='r', linestyle='--', label='Threshold (0.3)')
    ax2.axvline(x=scores.mean(), color='g', linestyle='--', label=f'Mean ({scores.mean():.3f})')
    ax2.set_xlabel('Grid Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of grid scores')
    ax2.legend()
    
    n_grid = np.sum(scores > 0.3)
    print(f'\nCells with grid score > 0.3: {n_grid} ({100*n_grid/len(scores):.1f}%)')
    
    plt.tight_layout()
    plt.show()
    
    return scores, ratemaps

def get_spectral_phase_colors(rate_maps, grid_scores, mask_threshold=0.1):
    colors = []
    
    for rmap, gscore in zip(rate_maps, grid_scores):
        if np.max(rmap) < mask_threshold:
            colors.append([0, 0, 0])
            continue

        # Compute 2D fourier transform
        f = fft2(rmap)
        fshift = fftshift(f) # Put low frequencies in center
        
        # Dominant frequency corresponds to grid periodicity
        center_y, center_x = fshift.shape[0]//2, fshift.shape[1]//2
        fshift[center_y, center_x] = 0 
        
        # Max power
        magnitude = np.abs(fshift)
        idx = np.argmax(magnitude)
        py, px = np.unravel_index(idx, magnitude.shape)
        
        # Extract phase at frequency
        phase_angle = np.angle(fshift[py, px]) 
        
        hue = (phase_angle + np.pi) / (2 * np.pi)
        
        sat = 1.0 
        val = 1.0
        
        colors.append(hsv_to_rgb([hue, sat, val]))
        
    return np.array(colors)

# Logging-helper analyze functions for during training

def compute_grid_scores_from_model(
        model, 
        data_gen, 
        n_batches=50, 
        batch_size=200, 
        seq_length=20, 
        res=32
    ):
    """
    Compute grid scores directly from a model and data generator
    
    Returns scores (array of grid scores for each cell) and ratemaps (array of ratemaps with size (n_cells, res, res))
    """
    model.eval()
    
    # Collect activations from short trajectories
    all_g = []
    all_pos = []
    
    for i in range(n_batches):
        velocity, init_pc, _, positions = data_gen.generate_batch(
            batch_size=batch_size,
            seq_length=seq_length
        )
        
        with torch.no_grad():
            g = model.g(velocity, init_pc)
        
        g_flat = g.reshape(-1, model.Ng).cpu().numpy()
        pos_flat = positions.reshape(-1, 2).cpu().numpy()
        
        all_g.append(g_flat)
        all_pos.append(pos_flat)
    
    all_g = np.concatenate(all_g, axis=0)
    all_pos = np.concatenate(all_pos, axis=0)
    
    num_cells = all_g.shape[1]
    
    scores = np.zeros(num_cells)
    ratemaps = np.zeros((num_cells, res, res))
    
    for i in range(num_cells):
        ratemap = get_ratemap(all_pos, all_g[:, i], res=res, 
                              box_width=data_gen.box_size, box_height=data_gen.box_size)
        ratemaps[i] = ratemap
        scores[i], _ = get_grid_score(ratemap)
    
    model.train()
    return scores, ratemaps

def create_top_ratemaps_figure(ratemaps, grid_scores, n_top=10, box_size=2.2):
    """
    Create a figure showing the top n ratemaps by grid score
    
    Returns matplotlib figure (should be closed after use)
    """
    sorted_idx = np.argsort(grid_scores)[::-1]
    top_idx = sorted_idx[:n_top]
    
    cols = min(5, n_top)
    rows = (n_top + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, idx in enumerate(top_idx):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        im = ax.imshow(ratemaps[idx], origin='lower', 
                       extent=[-box_size/2, box_size/2, -box_size/2, box_size/2],
                       cmap='jet')
        ax.set_title(f'Cell {idx}\nScore: {grid_scores[idx]:.3f}', fontsize=9)
        ax.axis('off')
    
    # Hide unused axes
    for i in range(n_top, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'Top {n_top} Grid Cells by Score', fontsize=12)
    plt.tight_layout()
    
    return fig

def create_grid_score_histogram_figure(grid_scores, threshold=0.3):
    """
    Create a histogram of grid scores
    
    Returns matplotlib figure (should be closed after use)
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(grid_scores, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    ax.axvline(x=grid_scores.mean(), color='g', linestyle='--', label=f'Mean ({grid_scores.mean():.3f})')
    ax.set_xlabel('Grid Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Grid Scores')
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_topographic_phase_map_figure(ratemaps, grid_scores, mask_threshold=0.0):
    """
    Create a topographic phase map visualization
    
    Returns matplotlib figure (should be closed after use)
    """
    colors = get_spectral_phase_colors(ratemaps, grid_scores, mask_threshold=mask_threshold)
    
    side = int(np.sqrt(len(colors)))
    topo_map = colors.reshape(side, side, 3)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(topo_map, interpolation='nearest')
    ax.set_title("Topographic Phase Map")
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_trajectory_decoding_figure(
        model,
        data_gen,
        n_trajectories=25,
        seq_length=20
    ):
    """
    Create a trajectory decoding visualization figure
    
    Returns matplotlib figure, mean position error in cm (matplotlib figure should be closed after use)
    """
    model.eval()
    
    velocity, init_pc, target_pc, positions = data_gen.generate_batch(
        batch_size=n_trajectories,
        seq_length=seq_length
    )

    with torch.no_grad():
        place_logits, _ = model(velocity, init_pc)
        place_cell_centers = data_gen.place_cells.us  
        pred_pos = model.decode_position(place_logits, place_cell_centers)

    pos = positions.cpu().numpy()
    pred_pos = pred_pos.cpu().numpy()
    us = data_gen.place_cells.us.cpu().numpy()

    box_size = data_gen.box_size

    # Calculate mean position error
    pos_error = np.sqrt(((pos - pred_pos) ** 2).sum(axis=-1))
    mean_error_cm = pos_error.mean() * 100

    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # Plot place cell centers in background
    ax.scatter(us[:, 0], us[:, 1], s=20, alpha=0.5, c='lightgrey')

    for i in range(n_trajectories):
        ax.plot(pos[:, i, 0], pos[:, i, 1], c='black', 
                label='True position' if i == 0 else None, linewidth=2)
        ax.plot(pred_pos[:, i, 0], pred_pos[:, i, 1], '.-', c='C1',
                label='Decoded position' if i == 0 else None)

    ax.legend()

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-box_size / 2, box_size / 2])
    ax.set_ylim([-box_size / 2, box_size / 2])
    ax.set_title(f'Trajectory Decoding (Mean Error: {mean_error_cm:.2f} cm)')

    plt.tight_layout()
    
    model.train()
    return fig, mean_error_cm
