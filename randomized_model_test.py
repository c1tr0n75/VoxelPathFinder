from pathfinding_nn import PathfindingNetwork, create_voxel_input
import torch, numpy as np
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = Path('training_outputs/final_model.pth')

# Load model and weights
model = PathfindingNetwork().to(device).eval()
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
# Build a random environment
voxel_dim = model.voxel_dim  # (32, 32, 32)
D, H, W = voxel_dim
obstacle_prob = 0.2

obstacles = (np.random.rand(D, H, W) < obstacle_prob).astype(np.float32)

# Pick random free start/goal
free = np.argwhere(obstacles == 0)
start_idx, goal_idx = np.random.choice(len(free), size=2, replace=False)
start = tuple(free[start_idx])
goal = tuple(free[goal_idx])

# Create model inputs
voxel_np = create_voxel_input(obstacles, start, goal, voxel_dim=voxel_dim)  # (3,32,32,32)
voxel = torch.from_numpy(voxel_np).float().unsqueeze(0).to(device)          # (1,3,32,32,32)
pos = torch.tensor([[start, goal]], dtype=torch.long, device=device)        # (1,2,3)

# Inference (same as before)
with torch.no_grad():
    actions = model(voxel, pos)[0].tolist()

print("start position : ", start)
print("goal position : ", goal)
#print("obstacles : ", obstacles)
print("actions taken : ", actions)