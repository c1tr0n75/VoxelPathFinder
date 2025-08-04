import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Import your network architecture (assuming it's in the same directory)
# from pathfinding_nn import PathfindingNetwork, PathfindingLoss

class SyntheticPathfindingDataset(Dataset):
    """
    PyTorch Dataset for loading synthetic pathfinding data.
    """
    def __init__(self, dataset_path="./sample_dataset", max_sequence_length=100):
        self.dataset_path = Path(dataset_path)
        self.max_sequence_length = max_sequence_length
        
        # Find all sample files
        self.sample_files = sorted(list(self.dataset_path.glob("sample_*.npz")))
        
        if len(self.sample_files) == 0:
            raise ValueError(f"No sample files found in {dataset_path}")
        
        print(f"📁 Loaded {len(self.sample_files)} training samples")
        
        # Analyze dataset statistics
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """Analyze dataset to understand sequence lengths and other stats."""
        path_lengths = []
        turn_counts = []
        
        for sample_file in self.sample_files[:100]:  # Sample first 100 for stats
            try:
                data = np.load(sample_file)
                path_lengths.append(int(data['path_length']))
                turn_counts.append(int(data['num_turns']))
            except:
                continue
        
        if path_lengths:
            print(f"📊 Dataset Statistics:")
            print(f"   Path lengths: {min(path_lengths)}-{max(path_lengths)} (avg: {np.mean(path_lengths):.1f})")
            print(f"   Turn counts: {min(turn_counts)}-{max(turn_counts)} (avg: {np.mean(turn_counts):.1f})")
            print(f"   Max sequence length set to: {self.max_sequence_length}")
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        sample_file = self.sample_files[idx]
        
        try:
            data = np.load(sample_file)
            
            voxel_data = data['voxel_data'].astype(np.float32)  # (3, 32, 32, 32)
            positions = data['positions'].astype(np.int64)      # (2, 3)
            target_actions = data['target_actions'].astype(np.int64)  # (seq_len,)
            path_length = int(data['path_length'])
            num_turns = int(data['num_turns'])
            
            # Pad or truncate action sequence
            if len(target_actions) > self.max_sequence_length:
                target_actions = target_actions[:self.max_sequence_length]
                path_length = self.max_sequence_length
            else:
                # Pad with -1 (padding token)
                padded_actions = np.full(self.max_sequence_length, -1, dtype=np.int64)
                padded_actions[:len(target_actions)] = target_actions
                target_actions = padded_actions
            
            return {
                'voxel_data': torch.FloatTensor(voxel_data),
                'positions': torch.LongTensor(positions),
                'target_actions': torch.LongTensor(target_actions),
                'path_length': path_length,
                'num_turns': num_turns,
                'sample_id': idx
            }
            
        except Exception as e:
            print(f"❌ Error loading sample {idx}: {e}")
            # Return a dummy sample
            return {
                'voxel_data': torch.zeros(3, 32, 32, 32),
                'positions': torch.zeros(2, 3, dtype=torch.long),
                'target_actions': torch.full((self.max_sequence_length,), -1, dtype=torch.long),
                'path_length': 0,
                'num_turns': 0,
                'sample_id': idx
            }


class PathfindingTrainer:
    """
    Complete training pipeline for the pathfinding network.
    """
    def __init__(self, 
                 model,
                 dataset_path="./sample_dataset",
                 batch_size=8,
                 learning_rate=1e-4,
                 max_sequence_length=100,
                 device=None):
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Model setup
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = PathfindingLoss(turn_penalty_weight=0.1, collision_penalty_weight=10.0)
        
        # Dataset setup
        self.dataset = SyntheticPathfindingDataset(dataset_path, max_sequence_length)
        
        # Split dataset (80% train, 20% validation)
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for debugging, increase for speed
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"📊 Training samples: {len(self.train_dataset)}")
        print(f"📊 Validation samples: {len(self.val_dataset)}")
        
        # Training history
        self.train_history = {
            'loss': [],
            'path_loss': [],
            'turn_loss': [],
            'collision_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_path_loss = 0
        total_turn_loss = 0
        total_collision_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            # Move batch to device
            voxel_data = batch['voxel_data'].to(self.device)
            positions = batch['positions'].to(self.device)
            target_actions = batch['target_actions'].to(self.device)
            
            # Filter out padding tokens (-1) for loss calculation
            valid_mask = target_actions != -1
            
            # Forward pass
            self.optimizer.zero_grad()
            action_logits, turn_penalties = self.model(voxel_data, positions, target_actions)
            
            # Calculate loss only on valid (non-padded) positions
            loss_dict = self.loss_fn(action_logits, turn_penalties, target_actions)
            
            # Apply mask to loss if needed (depends on your loss function implementation)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_path_loss += loss_dict['path_loss'].item()
            total_turn_loss += loss_dict['turn_loss'].item()
            total_collision_loss += loss_dict['collision_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'path': f"{loss_dict['path_loss'].item():.4f}",
                'turn': f"{loss_dict['turn_loss'].item():.4f}"
            })
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_path_loss = total_path_loss / num_batches
        avg_turn_loss = total_turn_loss / num_batches
        avg_collision_loss = total_collision_loss / num_batches
        
        return avg_loss, avg_path_loss, avg_turn_loss, avg_collision_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                voxel_data = batch['voxel_data'].to(self.device)
                positions = batch['positions'].to(self.device)
                target_actions = batch['target_actions'].to(self.device)
                
                # Forward pass
                action_logits, turn_penalties = self.model(voxel_data, positions, target_actions)
                
                # Calculate loss
                loss_dict = self.loss_fn(action_logits, turn_penalties, target_actions)
                total_loss += loss_dict['total_loss'].item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs=50, save_every=10):
        """Complete training loop."""
        print(f"\n🚀 Starting training for {num_epochs} epochs")
        print("=" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, path_loss, turn_loss, collision_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update history
            self.train_history['loss'].append(train_loss)
            self.train_history['path_loss'].append(path_loss)
            self.train_history['turn_loss'].append(turn_loss)
            self.train_history['collision_loss'].append(collision_loss)
            self.train_history['val_loss'].append(val_loss)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Path: {path_loss:.4f} | Turn: {turn_loss:.4f} | Collision: {collision_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pth")
                print(f"💾 New best model saved! Val Loss: {val_loss:.4f}")
            
            # Periodic save
            if (epoch + 1) % save_every == 0:
                self.save_model(f"model_epoch_{epoch + 1}.pth")
                self.save_training_history()
        
        print(f"\n✅ Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def save_model(self, filename):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history
        }, filename)
    
    def load_model(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        print(f"📂 Model loaded from {filename}")
    
    def save_training_history(self):
        """Save training history as JSON."""
        with open('training_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.train_history['loss']) + 1)
        
        # Total loss
        ax1.plot(epochs, self.train_history['loss'], 'b-', label='Train')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Validation')
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Path loss
        ax2.plot(epochs, self.train_history['path_loss'], 'g-')
        ax2.set_title('Path Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        # Turn loss
        ax3.plot(epochs, self.train_history['turn_loss'], 'm-')
        ax3.set_title('Turn Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        # Collision loss
        ax4.plot(epochs, self.train_history['collision_loss'], 'c-')
        ax4.set_title('Collision Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


# Example usage and quick test
if __name__ == "__main__":
    # Test dataset loading
    print("🧪 Testing dataset loading...")
    
    try:
        # Create a small dataset for testing
        dataset = SyntheticPathfindingDataset("./sample_dataset", max_sequence_length=50)
        
        # Test loading a sample
        sample = dataset[0]
        print(f"✅ Sample loaded successfully!")
        print(f"   Voxel data shape: {sample['voxel_data'].shape}")
        print(f"   Positions shape: {sample['positions'].shape}")
        print(f"   Target actions shape: {sample['target_actions'].shape}")
        print(f"   Path length: {sample['path_length']}")
        print(f"   Number of turns: {sample['num_turns']}")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(dataloader))
        print(f"✅ Batch loaded successfully!")
        print(f"   Batch voxel data shape: {batch['voxel_data'].shape}")
        print(f"   Batch positions shape: {batch['positions'].shape}")
        print(f"   Batch target actions shape: {batch['target_actions'].shape}")
        
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        print("Please ensure your dataset is in './sample_dataset' directory")

    print(f"\n🎯 Next Steps:")
    print("1. Run the dataset inspector to verify your data")
    print("2. Import your PathfindingNetwork and PathfindingLoss")
    print("3. Initialize the trainer and start training!")
    print("\nExample:")
    print("```python")
    print("# Import your network")
    print("from pathfinding_nn import PathfindingNetwork, PathfindingLoss")
    print("")
    print("# Create model")
    print("model = PathfindingNetwork()")
    print("")
    print("# Create trainer")
    print("trainer = PathfindingTrainer(model, dataset_path='./sample_dataset')")
    print("")
    print("# Start training")
    print("trainer.train(num_epochs=100)")
    print("```")
