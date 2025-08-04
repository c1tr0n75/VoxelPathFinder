"""
Main training script for 3D Pathfinding Neural Network

This script integrates your neural network architecture with the synthetic dataset
and provides a complete training pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add the current directory to path to import your modules
sys.path.append('.')

# Import your neural network architecture
try:
    from pathfinding_nn import PathfindingNetwork, PathfindingLoss
    print("✅ Successfully imported PathfindingNetwork and PathfindingLoss")
except ImportError as e:
    print(f"❌ Error importing neural network modules: {e}")
    print("Please ensure '3DPathPlanning_nn.py' is in the current directory")
    sys.exit(1)

# Import the training pipeline (assuming it's saved as training_pipeline.py)
try:
    from training_pipeline import PathfindingTrainer, SyntheticPathfindingDataset
    print("✅ Successfully imported training pipeline")
except ImportError as e:
    print(f"❌ Error importing training pipeline: {e}")
    print("Please ensure the training pipeline code is available")
    sys.exit(1)


def create_model_config():
    """Create model configuration."""
    config = {
        'voxel_dim': (32, 32, 32),
        'input_channels': 3,  # obstacles + start + goal channels
        'env_feature_dim': 512,
        'pos_feature_dim': 64,
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'max_sequence_length': 100,
        'num_actions': 6  # Forward, Back, Left, Right, Up, Down
    }
    return config


def setup_training_config():
    """Setup training configuration."""
    config = {
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'save_every': 10,
        'dataset_path': './sample_dataset',
        'output_dir': './training_outputs',
        'device': 'auto'  # auto, cpu, cuda
    }
    return config


def main():
    parser = argparse.ArgumentParser(description='Train 3D Pathfinding Neural Network')
    parser.add_argument('--dataset-path', type=str, default='./sample_dataset',
                        help='Path to synthetic dataset directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test dataset loading without training')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path('./training_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🚀 3D Pathfinding Neural Network Training")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Device: {device}")
    print(f"📁 Dataset path: {args.dataset_path}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print(f"🔄 Epochs: {args.epochs}")
    
    # Test dataset loading first
    print("\n📋 Testing dataset loading...")
    try:
        dataset = SyntheticPathfindingDataset(args.dataset_path, max_sequence_length=100)
        print(f"✅ Dataset loaded successfully: {len(dataset)} samples")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"✅ Sample structure verified:")
        print(f"   Voxel data: {sample['voxel_data'].shape}")
        print(f"   Positions: {sample['positions'].shape}")
        print(f"   Actions: {sample['target_actions'].shape}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    if args.test_only:
        print("✅ Dataset test completed successfully!")
        return
    
    # Create model
    print("\n🏗️ Creating model...")
    model_config = create_model_config()
    
    try:
        model = PathfindingNetwork(
            voxel_dim=model_config['voxel_dim'],
            input_channels=model_config['input_channels'],
            env_feature_dim=model_config['env_feature_dim'],
            pos_feature_dim=model_config['pos_feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_actions=model_config['num_actions']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    # Create trainer
    print("\n🎯 Setting up trainer...")
    try:
        trainer = PathfindingTrainer(
            model=model,
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_sequence_length=100,
            device=device
        )
        
        print("✅ Trainer setup completed")
        
    except Exception as e:
        print(f"❌ Trainer setup failed: {e}")
        return
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        try:
            trainer.load_model(args.resume)
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return
    
    # Save configuration
    config_info = {
        'model_config': model_config,
        'training_config': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.epochs,
            'dataset_path': args.dataset_path,
            'device': str(device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        },
        'dataset_info': {
            'total_samples': len(dataset),
            'train_samples': len(trainer.train_dataset),
            'val_samples': len(trainer.val_dataset)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(config_info, f, indent=2)
    
    # Start training
    print("\n🏃 Starting training...")
    print("=" * 60)
    
    try:
        trainer.train(num_epochs=args.epochs, save_every=10)
        
        # Save final results
        print("\n💾 Saving final results...")
        trainer.save_model(output_dir / 'final_model.pth')
        trainer.save_training_history()
        
        # Plot training curves
        print("📊 Generating training plots...")
        trainer.plot_training_curves()
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        print("💾 Saving current progress...")
        trainer.save_model(output_dir / 'interrupted_model.pth')
        trainer.save_training_history()
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("💾 Attempting to save current progress...")
        try:
            trainer.save_model(output_dir / 'error_model.pth')
            trainer.save_training_history()
        except:
            print("❌ Could not save progress")


def test_model_inference():
    """Test trained model inference on a sample."""
    print("\n🧪 Testing model inference...")
    
    try:
        # Load a trained model (adjust path as needed)
        model_config = create_model_config()
        model = PathfindingNetwork(**{k: v for k, v in model_config.items() 
                                   if k in ['voxel_dim', 'input_channels', 'env_feature_dim', 
                                           'pos_feature_dim', 'hidden_dim', 'num_actions']})
        
        # Load dataset
        dataset = SyntheticPathfindingDataset('./sample_dataset', max_sequence_length=100)
        sample = dataset[0]
        
        # Prepare input (add batch dimension)
        voxel_data = sample['voxel_data'].unsqueeze(0)  # (1, 3, 32, 32, 32)
        positions = sample['positions'].unsqueeze(0)    # (1, 2, 3)
        
        print(f"Input shapes:")
        print(f"  Voxel data: {voxel_data.shape}")
        print(f"  Positions: {positions.shape}")
        print(f"  Ground truth actions: {sample['target_actions'][:10]}...")
        
        # Inference
        model.eval()
        with torch.no_grad():
            generated_path = model(voxel_data, positions)
        
        print(f"Generated path: {generated_path[0][:10]}...")
        print(f"Generated path shape: {generated_path.shape}")
        
        # Decode actions
        action_names = ['FORWARD', 'BACK', 'LEFT', 'RIGHT', 'UP', 'DOWN']
        gt_actions = [action_names[a] for a in sample['target_actions'][:10] if a >= 0]
        gen_actions = [action_names[a] for a in generated_path[0][:10]]
        
        print(f"Ground truth: {gt_actions}")
        print(f"Generated: {gen_actions}")
        
        print("✅ Inference test completed!")
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")


def quick_dataset_stats():
    """Generate quick statistics about the dataset."""
    print("\n📊 Generating dataset statistics...")
    
    try:
        dataset_path = Path('./sample_dataset')
        sample_files = list(dataset_path.glob('sample_*.npz'))
        
        if not sample_files:
            print("❌ No dataset files found")
            return
        
        stats = {
            'path_lengths': [],
            'turn_counts': [],
            'start_positions': [],
            'goal_positions': [],
            'obstacle_densities': []
        }
        
        print(f"Analyzing {len(sample_files)} samples...")
        
        for sample_file in sample_files[:100]:  # Analyze first 100 samples
            try:
                data = np.load(sample_file)
                
                voxel_data = data['voxel_data']
                positions = data['positions']
                path_length = int(data['path_length'])
                num_turns = int(data['num_turns'])
                
                # Calculate obstacle density
                obstacles = voxel_data[0]  # First channel is obstacles
                total_voxels = obstacles.size
                obstacle_voxels = np.sum(obstacles)
                density = obstacle_voxels / total_voxels
                
                stats['path_lengths'].append(path_length)
                stats['turn_counts'].append(num_turns)
                stats['start_positions'].append(positions[0])
                stats['goal_positions'].append(positions[1])
                stats['obstacle_densities'].append(density)
                
            except Exception as e:
                print(f"Warning: Could not process {sample_file}: {e}")
                continue
        
        # Print statistics
        if stats['path_lengths']:
            print(f"\n📈 Dataset Statistics:")
            print(f"   Samples analyzed: {len(stats['path_lengths'])}")
            print(f"   Path lengths: min={min(stats['path_lengths'])}, max={max(stats['path_lengths'])}, avg={np.mean(stats['path_lengths']):.1f}")
            print(f"   Turn counts: min={min(stats['turn_counts'])}, max={max(stats['turn_counts'])}, avg={np.mean(stats['turn_counts']):.1f}")
            print(f"   Obstacle density: min={min(stats['obstacle_densities']):.3f}, max={max(stats['obstacle_densities']):.3f}, avg={np.mean(stats['obstacle_densities']):.3f}")
            
            # Turn efficiency
            turn_efficiency = [turns/length if length > 0 else 0 for turns, length in zip(stats['turn_counts'], stats['path_lengths'])]
            print(f"   Turn efficiency (turns/length): avg={np.mean(turn_efficiency):.3f}")
        
        print("✅ Dataset analysis completed!")
        
    except Exception as e:
        print(f"❌ Dataset analysis failed: {e}")


if __name__ == "__main__":
    # You can also run specific functions for testing
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--stats':
        quick_dataset_stats()
    elif len(sys.argv) > 1 and sys.argv[1] == '--test-inference':
        test_model_inference()
    else:
        main()