import os
import time

import torch
import torch.optim as optim
from datagen import DataGenerator
from model import GridCellNet


# Parameters
TOTAL_STEPS = 30000
PRINT_EVERY = 1000
SAVE_EVERY = 5000

BATCH_SIZE = 64
SEQ_LENGTH = 100

LEARNING_RATE = 2e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1e-5

SAVE_DIR = './checkpoints/'
os.makedirs(SAVE_DIR, exist_ok=True)

def train():
    generator = DataGenerator()

    model = GridCellNet()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    #optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_time = time.time()
    model.train()

    for step in range(1, TOTAL_STEPS + 1):
        inputs, target_pc, target_hd = generator.generate_batch()

        inputs = inputs.to(device)
        target_pc = target_pc.to(device)
        target_hd = target_hd.to(device)

        optimizer.zero_grad()
        pc_logits, hd_logits, _, _ = model(inputs)

        log_probs_pc = torch.nn.functional.log_softmax(pc_logits, dim=-1)
        log_probs_hd = torch.nn.functional.log_softmax(hd_logits, dim=-1)

        loss_pc = -torch.sum(target_pc * log_probs_pc, dim=1).mean()
        loss_hd = -torch.sum(target_hd * log_probs_hd, dim=1).mean()

        total_loss = loss_pc + loss_hd

        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), GRADIENT_CLIP)

        optimizer.step()

        if step % PRINT_EVERY == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{TOTAL_STEPS} | "
                  f"Loss: {total_loss.item():.4f} (PC: {loss_pc.item():.4f}, HD: {loss_hd.item():.4f}) | "
                  f"Time: {elapsed:.1f}s")
            
        if step % SAVE_EVERY == 0:
            save_path = os.path.join(SAVE_DIR, f"model_step_{step}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"-> Saved checkpoint: {save_path}")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model_final.pth"))
    print("Training Complete")

if __name__ == '__main__':
    train()