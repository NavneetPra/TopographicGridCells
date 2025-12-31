import numpy as np
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, HeadDirectionCells
import torch


class DataGenerator:
    def __init__(self):
        self.env = Environment(params={'boundary': [[0,0], [2.2,0], [2.2,2.2], [0,2.2]]})
        self.agent = Agent(self.env)

        self.pcs = PlaceCells(self.agent, params={'n': 256, 'widths': 0.2})
        self.hdcs = HeadDirectionCells(self.agent, params={'n': 12})

    def generate_batch(self, batch_size=10, seq_length=100):
        inputs = []
        target_pcs = []
        target_hdcs = []

        for _ in range(batch_size):
            self.agent.reset_history()

            random_pos = self.env.sample_positions(n=1, method='uniform')[0]
            random_head = np.random.uniform(0, 2*np.pi)
            self.agent.pos = random_pos
            self.agent.head_direction = random_head
            
            seq_inputs = []
            seq_pc = []
            seq_hd = []

            for _ in range(seq_length):
                self.agent.update()
                vel = self.agent.velocity
                v = np.linalg.norm(vel)
                rot = self.agent.rotational_velocity

                inp = [v, np.sin(rot), np.cos(rot)]
                
                pc_activity = self.pcs.get_state()
                hd_activity = self.hdcs.get_state()

                pc_activity = pc_activity / (np.sum(pc_activity) + 1e-10)
                hd_activity = hd_activity / (np.sum(hd_activity) + 1e-10)

                seq_inputs.append(inp)
                seq_pc.append(pc_activity.ravel())
                seq_hd.append(hd_activity.ravel())

            inputs.append(seq_inputs)
            target_pcs.append(seq_pc)
            target_hdcs.append(seq_hd)

        return (torch.tensor(np.array(inputs), dtype=torch.float32),
            torch.tensor(np.array(target_pcs), dtype=torch.float32),
            torch.tensor(np.array(target_hdcs), dtype=torch.float32))