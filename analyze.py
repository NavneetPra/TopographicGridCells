import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
import scipy.ndimage

from model import GridCellNet


MODEL_PATH = './checkpoints/model_final.pth'
OUTPUT_FOLDER = "./grid_plots"
DURATION_MINUTES = 30
DT = 0.02
STEPS = int((DURATION_MINUTES * 60) / DT)
BINS = 32

def get_ratemap(pos_history, activations, res=32, env_size=2.2):
    edges = np.linspace(0, env_size, res + 1)

    occupancy, _, _ = np.histogram2d(pos_history[:,0], pos_history[:,1], bins=edges)
    activity, _, _ = np.histogram2d(pos_history[:,0], pos_history[:,1], bins=edges, weights=activations)

    ratemap = np.divide(activity, occupancy, out=np.zeros_like(activity), where=occupancy!=0)

    return ratemap.T

def visualize_model(model_path=MODEL_PATH, output_folder=OUTPUT_FOLDER, save_imgs=True):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving plots to: {os.path.abspath(output_folder)}")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = GridCellNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print('Model loaded and set to eval')

    print(f"Simulating {DURATION_MINUTES} minute time")
    env = Environment(params={'boundary': [[0,0], [2.2,0], [2.2,2.2], [0,2.2]]})
    agent = Agent(env, params={'dt': DT})

    inputs = []
    positions = []

    for _ in range(STEPS):
        agent.update()
        positions.append(agent.pos.copy())

        vel = agent.velocity
        v = np.linalg.norm(vel)
        rot = agent.rotational_velocity
        inp = [v, np.sin(rot), np.cos(rot)]
        inputs.append(inp)

    inputs = torch.tensor([inputs], dtype=torch.float32).to(device)
    positions = np.array(positions)

    with torch.no_grad():
        _, _, _, g_activations = model(inputs)

    g_activations = g_activations.squeeze(0).cpu().numpy()
    num_units = g_activations.shape[1]

    print("Computing ratemaps and plotting")

    if save_imgs:
        for unit_idx in range(num_units):
            plt.figure(figsize=(4,4))

            activation_trace = g_activations[:, unit_idx]
            ratemap = get_ratemap(positions, activation_trace, res=BINS)

            ratemap_smooth = scipy.ndimage.gaussian_filter(ratemap, sigma=1.0)

            plt.imshow(ratemap_smooth, origin='lower', cmap='jet', interpolation='bilinear')
            plt.title(f'Unit {unit_idx}')
            plt.axis('off')

            filename = os.path.join(output_folder, f'unit_{unit_idx:03d}_ratemap.png')
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

            if unit_idx % 50 == 0:
                print(f"Saved {unit_idx}/{num_units}")

    num_plots = 16
    cols = 4
    rows = 4

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle('Linear Layer Activations')

    for i in range(num_plots):
        ax = axes[i // cols, i % cols]

        unit_idx = i
        activation_trace = g_activations[:, unit_idx]

        ratemap = get_ratemap(positions, activation_trace, res=BINS)

        ratemap_smooth = scipy.ndimage.gaussian_filter(ratemap, sigma=1.0)

        im = ax.imshow(ratemap_smooth, origin='lower', cmap='jet', interpolation='bilinear')
        ax.set_title(f'Unit {unit_idx}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    print('Done plotting')
