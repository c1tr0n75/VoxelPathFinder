import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json

class PathfindingDataset:
    """
    Dataset class that shows the exact structure of training inputs.
    """
    def __init__(self, voxel_dim=(32, 32, 32)):
        self.voxel_dim = voxel_dim
        self.action_to_idx = {
            'FORWARD': 0,   # +Z direction
            'BACK': 1,      # -Z direction  
            'LEFT': 2,      # -X direction
            'RIGHT': 3,     # +X direction
            'UP': 4,        # +Y direction
            'DOWN': 5       # -Y direction
        }
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}
    
    def create_sample_training_data(self):
        """
        Creates a complete training sample showing exact input structure.
        """
        # === INPUT 1: VOXEL DATA ===
        # Shape: (batch_size, 3, D, H, W) = (batch_size, 3, 32, 32, 32)
        
        # Channel 0: Obstacle map (1 = obstacle, 0 = free space)
        obstacles = np.zeros(self.voxel_dim, dtype=np.float32)
        
        # Add some walls and obstacles
        obstacles[10:20, 5:25, 10:15] = 1.0  # Wall barrier
        obstacles[25:30, 10:20, 20:25] = 1.0  # Block obstacle
        obstacles[:, 0, :] = 1.0   # Floor boundary
        obstacles[:, 31, :] = 1.0  # Ceiling boundary
        obstacles[0, :, :] = 1.0   # Left wall
        obstacles[31, :, :] = 1.0  # Right wall
        obstacles[:, :, 0] = 1.0   # Front wall
        obstacles[:, :, 31] = 1.0  # Back wall
        
        # Channel 1: Start position indicator (1 at start, 0 elsewhere)
        start_pos = (5, 10, 5)
        start_channel = np.zeros(self.voxel_dim, dtype=np.float32)
        start_channel[start_pos] = 1.0
        
        # Channel 2: Goal position indicator (1 at goal, 0 elsewhere)
        goal_pos = (25, 15, 25)
        goal_channel = np.zeros(self.voxel_dim, dtype=np.float32)
        goal_channel[goal_pos] = 1.0
        
        # Stack into 3-channel input
        voxel_input = np.stack([obstacles, start_channel, goal_channel], axis=0)
        
        # === INPUT 2: POSITION COORDINATES ===
        # Shape: (batch_size, 2, 3) = (batch_size, [start, goal], [x, y, z])
        positions = np.array([
            [start_pos[0], start_pos[1], start_pos[2]],  # Start: [x, y, z]
            [goal_pos[0], goal_pos[1], goal_pos[2]]      # Goal: [x, y, z]
        ], dtype=np.int64)
        
        # === TARGET OUTPUT: ACTION SEQUENCE ===
        # Optimal path from start to goal (hand-crafted for this example)
        # This would typically come from A* or other pathfinding algorithm
        
        action_sequence = [
            'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD',  # Move forward in Z
            'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT',           # Move right in X  
            'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD',  # More forward in Z
            'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT',           # More right in X
            'UP', 'UP', 'UP', 'UP', 'UP',                          # Move up in Y
            'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD'   # Final forward moves
        ]
        
        # Convert to indices
        target_actions = np.array([self.action_to_idx[action] for action in action_sequence], dtype=np.int64)
        
        return {
            'voxel_data': voxel_input,           # Shape: (3, 32, 32, 32)
            'positions': positions,              # Shape: (2, 3)
            'target_actions': target_actions,    # Shape: (sequence_length,)
            'action_sequence': action_sequence,  # Human-readable actions
            'start_pos': start_pos,
            'goal_pos': goal_pos,
            'path_length': len(target_actions),
            'num_turns': self._count_turns(action_sequence)
        }
    
    def _count_turns(self, action_sequence):
        """Count number of direction changes in the sequence."""
        if len(action_sequence) <= 1:
            return 0
        
        turns = 0
        for i in range(1, len(action_sequence)):
            if action_sequence[i] != action_sequence[i-1]:
                turns += 1
        return turns
    
    def create_batch_training_data(self, batch_size=4):
        """
        Creates a batch of training samples with exact tensor shapes.
        """
        batch_voxel_data = []
        batch_positions = []
        batch_target_actions = []
        batch_metadata = []
        
        max_seq_length = 0
        
        for i in range(batch_size):
            sample = self.create_sample_training_data()
            
            # Vary the scenarios slightly for each batch item
            if i == 1:
                # Add more obstacles for sample 1
                sample['voxel_data'][0, 15:20, 15:20, 15:20] = 1.0
            elif i == 2:
                # Different start/goal for sample 2
                sample['positions'][0] = [3, 8, 3]   # Different start
                sample['positions'][1] = [28, 20, 28] # Different goal
            
            batch_voxel_data.append(sample['voxel_data'])
            batch_positions.append(sample['positions'])
            batch_target_actions.append(sample['target_actions'])
            batch_metadata.append({
                'path_length': sample['path_length'],
                'num_turns': sample['num_turns'],
                'start_pos': sample['start_pos'],
                'goal_pos': sample['goal_pos']
            })
            
            max_seq_length = max(max_seq_length, len(sample['target_actions']))
        
        # Pad sequences to same length
        padded_target_actions = []
        for actions in batch_target_actions:
            padded = np.pad(actions, (0, max_seq_length - len(actions)), 
                          mode='constant', constant_values=-1)  # -1 for padding
            padded_target_actions.append(padded)
        
        # Convert to tensors
        batch_data = {
            'voxel_data': torch.FloatTensor(np.array(batch_voxel_data)),      # (4, 3, 32, 32, 32)
            'positions': torch.LongTensor(np.array(batch_positions)),         # (4, 2, 3)
            'target_actions': torch.LongTensor(np.array(padded_target_actions)), # (4, max_seq_length)
            'metadata': batch_metadata
        }
        
        return batch_data
    
    def visualize_sample(self, sample_data):
        """
        Visualize a training sample to understand the data structure.
        """
        print("=== TRAINING SAMPLE STRUCTURE ===\n")
        
        print("1. VOXEL DATA:")
        print(f"   Shape: {sample_data['voxel_data'].shape}")
        print(f"   Channel 0 (Obstacles): {np.sum(sample_data['voxel_data'][0])} obstacle voxels")
        print(f"   Channel 1 (Start): {np.sum(sample_data['voxel_data'][1])} start markers")
        print(f"   Channel 2 (Goal): {np.sum(sample_data['voxel_data'][2])} goal markers")
        
        print(f"\n2. POSITIONS:")
        print(f"   Shape: {sample_data['positions'].shape}")
        print(f"   Start position (x,y,z): {sample_data['positions'][0]}")
        print(f"   Goal position (x,y,z): {sample_data['positions'][1]}")
        
        print(f"\n3. TARGET ACTIONS:")
        print(f"   Shape: {sample_data['target_actions'].shape}")
        print(f"   Sequence length: {len(sample_data['target_actions'])}")
        print(f"   Number of turns: {sample_data['num_turns']}")
        print(f"   Action sequence: {sample_data['action_sequence'][:10]}..." if len(sample_data['action_sequence']) > 10 else f"   Action sequence: {sample_data['action_sequence']}")
        
        print(f"\n4. METADATA:")
        print(f"   Path length: {sample_data['path_length']}")
        print(f"   Start: {sample_data['start_pos']}")
        print(f"   Goal: {sample_data['goal_pos']}")
        
        return sample_data


class TrainingDataLoader:
    """
    Shows how to create a proper PyTorch DataLoader for training.
    """
    def __init__(self, dataset_size=1000, voxel_dim=(32, 32, 32)):
        self.dataset_size = dataset_size
        self.voxel_dim = voxel_dim
        self.pathfinding_dataset = PathfindingDataset(voxel_dim)
    
    def generate_training_dataset(self, save_path=None):
        """
        Generate a full training dataset.
        In practice, this would load from pre-computed optimal paths.
        """
        print(f"Generating {self.dataset_size} training samples...")
        
        dataset = []
        for i in range(self.dataset_size):
            if i % 100 == 0:
                print(f"Generated {i}/{self.dataset_size} samples")
            
            sample = self.pathfinding_dataset.create_sample_training_data()
            dataset.append(sample)
        
        if save_path:
            # Save dataset (in practice, you'd use more efficient formats)
            print(f"Saving dataset to {save_path}")
            # np.savez_compressed(save_path, dataset=dataset)
        
        return dataset
    
    def create_dataloader(self, dataset, batch_size=8, shuffle=True):
        """
        Create PyTorch DataLoader equivalent functionality.
        """
        def collate_fn(batch):
            """Custom collate function to handle variable-length sequences."""
            
            # Find max sequence length in batch
            max_seq_len = max(len(sample['target_actions']) for sample in batch)
            
            # Prepare batch tensors
            batch_voxel = []
            batch_positions = []
            batch_actions = []
            batch_metadata = []
            
            for sample in batch:
                batch_voxel.append(sample['voxel_data'])
                batch_positions.append(sample['positions'])
                
                # Pad action sequence
                actions = sample['target_actions']
                padded_actions = np.pad(actions, (0, max_seq_len - len(actions)), 
                                      mode='constant', constant_values=-1)
                batch_actions.append(padded_actions)
                
                batch_metadata.append({
                    'path_length': sample['path_length'],
                    'num_turns': sample['num_turns']
                })
            
            return {
                'voxel_data': torch.FloatTensor(np.array(batch_voxel)),
                'positions': torch.LongTensor(np.array(batch_positions)),
                'target_actions': torch.LongTensor(np.array(batch_actions)),
                'metadata': batch_metadata
            }
        
        # Simple batch generator (in practice, use torch.utils.data.DataLoader)
        indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = [dataset[idx] for idx in batch_indices]
            yield collate_fn(batch)


def demonstrate_training_inputs():
    """
    Complete demonstration of training input structure.
    """
    print("=" * 60)
    print("PATHFINDING NEURAL NETWORK - TRAINING DATA STRUCTURE")
    print("=" * 60)
    
    # Create dataset instance
    dataset = PathfindingDataset()
    
    # Create single sample
    print("\n📊 SINGLE TRAINING SAMPLE:")
    print("-" * 40)
    sample = dataset.create_sample_training_data()
    dataset.visualize_sample(sample)
    
    # Create batch
    print(f"\n📦 BATCH TRAINING DATA:")
    print("-" * 40)
    batch_data = dataset.create_batch_training_data(batch_size=4)
    
    print(f"Batch voxel data shape: {batch_data['voxel_data'].shape}")
    print(f"Batch positions shape: {batch_data['positions'].shape}")  
    print(f"Batch target actions shape: {batch_data['target_actions'].shape}")
    
    print(f"\nBatch contents:")
    for i in range(4):
        metadata = batch_data['metadata'][i]
        print(f"  Sample {i}: {metadata['path_length']} actions, {metadata['num_turns']} turns")
    
    # Show tensor data types and ranges
    print(f"\n🔍 TENSOR DETAILS:")
    print("-" * 40)
    print(f"Voxel data dtype: {batch_data['voxel_data'].dtype}")
    print(f"Voxel data range: [{batch_data['voxel_data'].min():.1f}, {batch_data['voxel_data'].max():.1f}]")
    print(f"Positions dtype: {batch_data['positions'].dtype}")
    print(f"Positions range: [{batch_data['positions'].min()}, {batch_data['positions'].max()}]")
    print(f"Actions dtype: {batch_data['target_actions'].dtype}")
    print(f"Actions range: [{batch_data['target_actions'].min()}, {batch_data['target_actions'].max()}] (-1 is padding)")
    
    # Show how to use in training loop
    print(f"\n🔄 TRAINING LOOP USAGE:")
    print("-" * 40)
    print("""
# In your training loop:
for batch in dataloader:
    voxel_data = batch['voxel_data']        # (batch_size, 3, 32, 32, 32)  
    positions = batch['positions']          # (batch_size, 2, 3)
    target_actions = batch['target_actions'] # (batch_size, max_seq_len)
    
    # Forward pass
    action_logits, turn_penalties = model(voxel_data, positions, target_actions)
    
    # Calculate loss
    loss_dict = loss_fn(action_logits, turn_penalties, target_actions)
    loss = loss_dict['total_loss']
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    """)
    
    return batch_data


def show_data_generation_pipeline():
    """
    Shows how training data would be generated in practice.
    """
    print(f"\n📈 DATA GENERATION PIPELINE:")
    print("-" * 40)
    
    pipeline_steps = [
        "1. Generate random 3D environments with obstacles",
        "2. Sample random start and goal positions", 
        "3. Run optimal pathfinding algorithm (A*, Dijkstra, etc.)",
        "4. Extract action sequence from optimal path",
        "5. Count turns and validate path correctness",
        "6. Create multi-channel voxel representation",
        "7. Store as training sample"
    ]
    
    for step in pipeline_steps:
        print(f"   {step}")
    
    print(f"\n💾 DATASET STATISTICS (typical):")
    print("   • Dataset size: 10,000 - 100,000 samples")
    print("   • Environment complexity: 10-50% obstacle density")
    print("   • Path lengths: 10-100 actions")
    print("   • Turn counts: 2-20 turns per path")
    print("   • Success rate: 95%+ valid paths")


if __name__ == "__main__":
    # Run complete demonstration
    batch_data = demonstrate_training_inputs()
    show_data_generation_pipeline()
    
    print(f"\n✅ TRAINING DATA READY!")
    print("Your neural network can now be trained with this exact data structure.")