import torch, numpy as np
from pathlib import Path
from pathfinding_nn import PathfindingNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = Path('training_outputs/final_model.pth')
sample_path = next(Path('sample_dataset').glob('sample_*.npz'))  # first sample

# Load model and weights
model = PathfindingNetwork().to(device).eval()
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])

# Load one sample
data = np.load(sample_path)
voxel = torch.from_numpy(data['voxel_data']).float().unsqueeze(0).to(device)  # (1,3,32,32,32)
pos = torch.from_numpy(data['positions']).long().unsqueeze(0).to(device)      # (1,2,3)

# Inference
with torch.no_grad():
    actions = model(voxel, pos)[0].tolist()  # list of token IDs

# Decode valid movement actions (0..5)
action_names = ['FORWARD','BACK','LEFT','RIGHT','UP','DOWN']
decoded = [action_names[a] for a in actions if 0 <= a < 6]

print(f'Device: {device}')
print(f'Sample: {sample_path.name}')
print(f'Generated {len(decoded)} actions (first 20): {decoded[:20]}')