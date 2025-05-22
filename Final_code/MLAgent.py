# MLAgent.py
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        out += identity
        return self.relu(out)

class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(12, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(5)])
        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 2048), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(2048, action_size)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.policy_head(x)

def load_model(path, device):
    chk = torch.load(path, map_location=device)
    move2idx = chk.get('move2idx')
    state_dict = chk.get('model', chk.get('model_state_dict'))
    model = PolicyNet(len(move2idx))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    idx2move = {i:m for m,i in move2idx.items()}
    return model, move2idx, idx2move

def encode_board(gs_board):
    cmap = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
            'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}
    tensor = torch.zeros(1,12,8,8, dtype=torch.float)
    for r in range(8):
        for c in range(8):
            sq = gs_board[r][c]
            if sq == '--': continue
            color, p = sq[0], sq[1]
            sym = p.upper() if color=='w' else p.lower()
            idx = cmap[sym]
            tensor[0, idx, r, c] = 1.0
    return tensor
