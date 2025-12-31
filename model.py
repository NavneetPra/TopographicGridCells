import torch.nn as nn


class GridCellNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=3, hidden_size=128, batch_first = True)

        self.linear = nn.Linear(128, 512)
        self.dropout = nn.Dropout(p=0.5)

        self.pc_decoder = nn.Linear(512, 256)
        self.hd_decoder = nn.Linear(512, 12)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)

        g = self.linear(lstm_out)
        g_dropped = self.dropout(g)

        pc_logits = self.pc_decoder(g_dropped)
        hd_logits = self.hd_decoder(g_dropped)

        return pc_logits, hd_logits, hidden, g