import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

def inspect_dataset(dataset_path="./sample_dataset"):
    """
    Inspect the generated synthetic dataset to verify structure and quality.
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path {dataset_path} not found!")
        return
    
    # Find all .npz files
    sample_files = list(dataset_path.glob("sample_*.npz"))
    print(f"📁 Found {len(sample_files)} training samples")
    
    if len(sample_files) == 0:
        print("❌ No sample files found!")
        return
    
    # Load first few samples to inspect structure
    sample_stats = {
        'path_lengths': [],
        'num_turns': [],
        'voxel_shapes': [],
        'position_shapes': [],
        'action_shapes': []
    }
    
    print("\n🔍 Inspecting first 5 samples:")
    print("-" * 50)
    
    for i, sample_file in enumerate(sample_files[:5]):
        try:
            data = np.load(sample_file)
            
            voxel_data = data['voxel_data']
            positions = data['positions'] 
            target_actions = data['target_actions']
            path_length = data['path_length']
            num_turns = data['num_turns']
            
            print(f"Sample {i}:")
            print(f"  Voxel data shape: {voxel_data.shape}")
            print(f"  Positions shape: {positions.shape}")
            print(f"  Actions shape: {target_actions.shape}")
            print(f"  Path length: {path_length}")
            print(f"  Number of turns: {num_turns}")
            print(f"  Start pos: {positions[0]}")
            print(f"  Goal pos: {positions[1]}")
            print()
            
            # Collect stats
            sample_stats['path_lengths'].append(path_length)
            sample_stats['num_turns'].append(num_turns)
            sample_stats['voxel_shapes'].append(voxel_data.shape)
            sample_stats['position_shapes'].append(positions.shape)
            sample_stats['action_shapes'].append(target_actions.shape)
            
        except Exception as e:
            print(f"❌ Error loading sample {i}: {e}")
    
    # Dataset statistics
    if sample_stats['path_lengths']:
        print("\n📊 Dataset Statistics:")
        print("-" * 50)
        print(f"Path length - Min: {min(sample_stats['path_lengths'])}, Max: {max(sample_stats['path_lengths'])}, Avg: {np.mean(sample_stats['path_lengths']):.1f}")
        print(f"Turn count - Min: {min(sample_stats['num_turns'])}, Max: {max(sample_stats['num_turns'])}, Avg: {np.mean(sample_stats['num_turns']):.1f}")
        print(f"All voxel shapes consistent: {len(set(sample_stats['voxel_shapes'])) == 1}")
        print(f"All position shapes consistent: {len(set(sample_stats['position_shapes'])) == 1}")
    
    return sample_stats

def visualize_sample(sample_file_path):
    """
    Visualize a single training sample to understand the data structure.
    """
    try:
        data = np.load(sample_file_path)
        
        voxel_data = data['voxel_data']  # Shape: (3, 32, 32, 32)
        positions = data['positions']    # Shape: (2, 3)
        target_actions = data['target_actions']
        
        # Extract channels
        obstacles = voxel_data[0]  # Obstacle channel
        start_channel = voxel_data[1]  # Start position channel
        goal_channel = voxel_data[2]   # Goal position channel
        
        print(f"📈 Visualizing sample: {sample_file_path}")
        print(f"Obstacle voxels: {np.sum(obstacles)} / {obstacles.size}")
        print(f"Start position: {positions[0]}")
        print(f"Goal position: {positions[1]}")
        print(f"Action sequence length: {len(target_actions)}")
        
        # Action mapping for readability
        action_names = ['FORWARD', 'BACK', 'LEFT', 'RIGHT', 'UP', 'DOWN']
        action_sequence = [action_names[action] for action in target_actions]
        print(f"Action sequence: {action_sequence[:10]}{'...' if len(action_sequence) > 10 else ''}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error visualizing sample: {e}")
        return False

if __name__ == "__main__":
    # Inspect the dataset
    stats = inspect_dataset("./sample_dataset")
    
    # Visualize first sample if available
    sample_files = list(Path("./sample_dataset").glob("sample_*.npz"))
    if sample_files:
        print("\n" + "="*60)
        visualize_sample(sample_files[0])
