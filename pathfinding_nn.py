import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

class VoxelCNNEncoder(nn.Module):
    """
    Enhanced 3D CNN encoder for voxelized obstruction data with multi-channel support.
    Processes environment obstacles, start position, and goal position.
    """
    def __init__(self,
                 input_channels=3,  # obstacles + start + goal
                 filters_1=32,
                 kernel_size_1=(3, 3, 3),
                 pool_size_1=(2, 2, 2),
                 filters_2=64,
                 kernel_size_2=(3, 3, 3),
                 pool_size_2=(2, 2, 2),
                 filters_3=128,
                 kernel_size_3=(3, 3, 3),
                 pool_size_3=(2, 2, 2),
                 dense_units=512,
                 input_voxel_dim=(32, 32, 32),
                 dropout_rate=0.2
                ):
        super(VoxelCNNEncoder, self).__init__()

        self.input_voxel_dim = input_voxel_dim
        self.input_channels = input_channels

        # First 3D Convolutional Block
        padding_1 = tuple([(k - 1) // 2 for k in kernel_size_1])
        self.conv1 = nn.Conv3d(input_channels, filters_1, kernel_size_1, padding=padding_1)
        self.bn1 = nn.BatchNorm3d(filters_1)
        self.pool1 = nn.MaxPool3d(pool_size_1)
        self.dropout1 = nn.Dropout3d(dropout_rate)

        # Second 3D Convolutional Block
        padding_2 = tuple([(k - 1) // 2 for k in kernel_size_2])
        self.conv2 = nn.Conv3d(filters_1, filters_2, kernel_size_2, padding=padding_2)
        self.bn2 = nn.BatchNorm3d(filters_2)
        self.pool2 = nn.MaxPool3d(pool_size_2)
        self.dropout2 = nn.Dropout3d(dropout_rate)

        # Third 3D Convolutional Block (for better feature extraction)
        padding_3 = tuple([(k - 1) // 2 for k in kernel_size_3])
        self.conv3 = nn.Conv3d(filters_2, filters_3, kernel_size_3, padding=padding_3)
        self.bn3 = nn.BatchNorm3d(filters_3)
        self.pool3 = nn.MaxPool3d(pool_size_3)
        self.dropout3 = nn.Dropout3d(dropout_rate)

        # Calculate flattened size
        self._to_linear_input_size = self._get_conv_output_size()

        # Dense layers with residual connection
        self.fc1 = nn.Linear(self._to_linear_input_size, dense_units)
        self.fc2 = nn.Linear(dense_units, dense_units)
        self.dropout_fc = nn.Dropout(dropout_rate)

    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, *self.input_voxel_dim)
            x = self.pool1(self.dropout1(self.bn1(F.relu(self.conv1(dummy_input)))))
            x = self.pool2(self.dropout2(self.bn2(F.relu(self.conv2(x)))))
            x = self.pool3(self.dropout3(self.bn3(F.relu(self.conv3(x)))))
            return x.numel()

    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second conv block
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third conv block
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten and dense layers
        x = x.view(x.size(0), -1)
        x1 = F.relu(self.fc1(x))
        x1 = self.dropout_fc(x1)
        x2 = F.relu(self.fc2(x1))
        
        # Residual connection
        return x1 + x2


class PositionEncoder(nn.Module):
    """
    Encodes start and goal positions with learned embeddings.
    """
    def __init__(self, voxel_dim=(32, 32, 32), position_embed_dim=64):
        super(PositionEncoder, self).__init__()
        self.voxel_dim = voxel_dim
        self.position_embed_dim = position_embed_dim
        
        # Learned position embeddings for each dimension
        self.x_embed = nn.Embedding(voxel_dim[0], position_embed_dim // 3)
        self.y_embed = nn.Embedding(voxel_dim[1], position_embed_dim // 3)
        self.z_embed = nn.Embedding(voxel_dim[2], position_embed_dim // 3)
        
        # Additional processing
        self.fc = nn.Linear(position_embed_dim, position_embed_dim)
        
    def forward(self, positions):
        """
        positions: (batch_size, 2, 3) - [start_pos, goal_pos] with (x, y, z)
        """
        batch_size = positions.size(0)
        
        # Extract coordinates
        x_coords = positions[:, :, 0].long()  # (batch_size, 2)
        y_coords = positions[:, :, 1].long()  # (batch_size, 2)
        z_coords = positions[:, :, 2].long()  # (batch_size, 2)
        
        # Get embeddings
        x_emb = self.x_embed(x_coords)  # (batch_size, 2, embed_dim//3)
        y_emb = self.y_embed(y_coords)  # (batch_size, 2, embed_dim//3)
        z_emb = self.z_embed(z_coords)  # (batch_size, 2, embed_dim//3)
        
        # Concatenate embeddings
        pos_emb = torch.cat([x_emb, y_emb, z_emb], dim=-1)  # (batch_size, 2, embed_dim)
        
        # Flatten start and goal embeddings
        pos_emb = pos_emb.view(batch_size, -1)  # (batch_size, 2 * embed_dim)
        
        return F.relu(self.fc(pos_emb))


class PathPlannerTransformer(nn.Module):
    """
    Transformer-based path planner that generates action sequences.
    """
    def __init__(self, 
                 env_feature_dim=512,
                 pos_feature_dim=64,
                 hidden_dim=256,
                 num_heads=8,
                 num_layers=4,
                 max_sequence_length=100,
                 num_actions=6):  # Forward, Back, Left, Right, Up, Down
        super(PathPlannerTransformer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.num_actions = num_actions
        
        # Feature fusion
        self.feature_fusion = nn.Linear(env_feature_dim + pos_feature_dim, hidden_dim)
        
        # Action embeddings
        self.action_embed = nn.Embedding(num_actions + 1, hidden_dim)  # +1 for start token
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_sequence_length, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_actions)
        
        # Turn penalty head (for minimizing turns)
        self.turn_penalty_head = nn.Linear(hidden_dim, 1)
        
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, env_features, pos_features, target_actions=None):
        """
        env_features: (batch_size, env_feature_dim)
        pos_features: (batch_size, pos_feature_dim)
        target_actions: (batch_size, seq_len) - for training
        """
        batch_size = env_features.size(0)
        
        # Fuse environment and position features
        fused_features = self.feature_fusion(torch.cat([env_features, pos_features], dim=1))
        
        # Create memory (encoder output) by repeating fused features
        memory = fused_features.unsqueeze(1).repeat(1, self.max_sequence_length, 1)
        
        if target_actions is not None:
            # Training mode: use teacher forcing
            seq_len = target_actions.size(1)
            
            # Create input sequence (start token + target_actions[:-1])
            start_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=target_actions.device)
            input_seq = torch.cat([start_tokens, target_actions[:, :-1]], dim=1)
            
            # Embed actions and add positional encoding
            embedded = self.action_embed(input_seq)
            embedded += self.pos_encoding[:, :seq_len, :].to(embedded.device)
            
            # Generate attention mask (causal mask)
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(embedded.device)
            
            # Transformer decoder forward pass
            output = self.transformer_decoder(
                tgt=embedded,
                memory=memory[:, :seq_len, :],
                tgt_mask=tgt_mask
            )
            
            # Output projections
            action_logits = self.output_proj(output)
            turn_penalties = self.turn_penalty_head(output)
            
            return action_logits, turn_penalties
        else:
            # Inference mode: generate sequence autoregressively
            return self._generate_path(memory, batch_size)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _generate_path(self, memory, batch_size):
        """Generate path sequence autoregressively"""
        device = memory.device
        generated_actions = []
        
        # Start with start token
        current_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for step in range(self.max_sequence_length):
            # Embed current sequence
            embedded = self.action_embed(current_input)
            seq_len = embedded.size(1)
            embedded += self.pos_encoding[:, :seq_len, :].to(device)
            
            # Generate causal mask
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(device)
            
            # Forward pass
            output = self.transformer_decoder(
                tgt=embedded,
                memory=memory[:, :seq_len, :],
                tgt_mask=tgt_mask
            )
            
            # Get next action probabilities
            next_action_logits = self.output_proj(output[:, -1, :])
            next_actions = torch.argmax(next_action_logits, dim=-1, keepdim=True)
            
            generated_actions.append(next_actions)
            
            # Update input sequence
            current_input = torch.cat([current_input, next_actions], dim=1)
            
            # Check for early stopping (could add end token logic here)
            
        return torch.cat(generated_actions, dim=1)


class PathfindingNetwork(nn.Module):
    """
    Complete pathfinding network combining CNN encoder, position encoder, and transformer planner.
    """
    def __init__(self, 
                 voxel_dim=(32, 32, 32),
                 input_channels=3,
                 env_feature_dim=512,
                 pos_feature_dim=64,
                 hidden_dim=256,
                 num_actions=6):
        super(PathfindingNetwork, self).__init__()
        
        self.voxel_encoder = VoxelCNNEncoder(
            input_channels=input_channels,
            dense_units=env_feature_dim,
            input_voxel_dim=voxel_dim
        )
        
        self.position_encoder = PositionEncoder(
            voxel_dim=voxel_dim,
            position_embed_dim=pos_feature_dim
        )
        
        self.path_planner = PathPlannerTransformer(
            env_feature_dim=env_feature_dim,
            pos_feature_dim=pos_feature_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions
        )
        
    def forward(self, voxel_data, positions, target_actions=None):
        """
        voxel_data: (batch_size, 3, D, H, W) - [obstacles, start_mask, goal_mask]
        positions: (batch_size, 2, 3) - [start_pos, goal_pos]
        target_actions: (batch_size, seq_len) - for training
        """
        # Encode environment
        env_features = self.voxel_encoder(voxel_data)
        
        # Encode positions
        pos_features = self.position_encoder(positions)
        
        # Generate path
        if target_actions is not None:
            action_logits, turn_penalties = self.path_planner(env_features, pos_features, target_actions)
            return action_logits, turn_penalties
        else:
            generated_path = self.path_planner(env_features, pos_features)
            return generated_path


class PathfindingLoss(nn.Module):
    """
    Custom loss function that balances path correctness and turn minimization.
    """
    def __init__(self, turn_penalty_weight=0.1, collision_penalty_weight=10.0):
        super(PathfindingLoss, self).__init__()
        self.turn_penalty_weight = turn_penalty_weight
        self.collision_penalty_weight = collision_penalty_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, action_logits, turn_penalties, target_actions, collision_mask=None):
        """
        action_logits: (batch_size, seq_len, num_actions)
        turn_penalties: (batch_size, seq_len, 1)
        target_actions: (batch_size, seq_len)
        collision_mask: (batch_size, seq_len) - 1 if collision, 0 if safe
        """
        batch_size, seq_len, num_actions = action_logits.shape
        
        # Reshape for cross entropy loss
        action_logits_flat = action_logits.view(-1, num_actions)
        target_actions_flat = target_actions.view(-1)
        
        # Path correctness loss
        path_loss = self.ce_loss(action_logits_flat, target_actions_flat)
        
        # Turn minimization loss
        turn_loss = turn_penalties.mean()
        
        # Collision penalty
        collision_loss = 0.0
        if collision_mask is not None:
            collision_loss = (collision_mask.float()).mean() * self.collision_penalty_weight
            
        total_loss = path_loss + self.turn_penalty_weight * turn_loss + collision_loss
        
        return {
            'total_loss': total_loss,
            'path_loss': path_loss,
            'turn_loss': turn_loss,
            'collision_loss': collision_loss
        }


# Utility functions for data preparation
def create_voxel_input(obstacles, start_pos, goal_pos, voxel_dim=(32, 32, 32)):
    """
    Create multi-channel voxel input.
    
    obstacles: (D, H, W) binary array
    start_pos: (x, y, z) tuple
    goal_pos: (x, y, z) tuple
    """
    # Channel 0: obstacles
    obstacle_channel = obstacles.astype(np.float32)
    
    # Channel 1: start position
    start_channel = np.zeros(voxel_dim, dtype=np.float32)
    start_channel[start_pos] = 1.0
    
    # Channel 2: goal position
    goal_channel = np.zeros(voxel_dim, dtype=np.float32)
    goal_channel[goal_pos] = 1.0
    
    # Stack channels
    voxel_input = np.stack([obstacle_channel, start_channel, goal_channel], axis=0)
    
    return voxel_input


# Example usage and testing
if __name__ == "__main__":
    # Define problem parameters
    voxel_dim = (32, 32, 32)
    batch_size = 4
    num_actions = 6  # Forward, Back, Left, Right, Up, Down
    
    # Create the complete pathfinding network
    pathfinding_net = PathfindingNetwork(
        voxel_dim=voxel_dim,
        input_channels=3,
        env_feature_dim=512,
        pos_feature_dim=64,
        hidden_dim=256,
        num_actions=num_actions
    )
    
    print("=== 3D Pathfinding Network Architecture ===")
    print(f"Total parameters: {sum(p.numel() for p in pathfinding_net.parameters()):,}")
    
    # Create dummy data
    dummy_voxel_data = torch.randn(batch_size, 3, *voxel_dim)
    dummy_positions = torch.randint(0, 32, (batch_size, 2, 3))  # start and goal positions
    dummy_target_actions = torch.randint(0, num_actions, (batch_size, 20))  # sequence of 20 actions
    
    print(f"\n=== Testing Forward Pass ===")
    print(f"Input voxel shape: {dummy_voxel_data.shape}")
    print(f"Input positions shape: {dummy_positions.shape}")
    print(f"Target actions shape: {dummy_target_actions.shape}")
    
    # Training forward pass
    pathfinding_net.train()
    action_logits, turn_penalties = pathfinding_net(
        dummy_voxel_data, 
        dummy_positions, 
        dummy_target_actions
    )
    
    print(f"\nTraining mode outputs:")
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Turn penalties shape: {turn_penalties.shape}")
    
    # Inference forward pass
    pathfinding_net.eval()
    with torch.no_grad():
        generated_paths = pathfinding_net(dummy_voxel_data, dummy_positions)
    
    print(f"\nInference mode outputs:")
    print(f"Generated paths shape: {generated_paths.shape}")
    
    # Test loss function
    loss_fn = PathfindingLoss(turn_penalty_weight=0.1)
    loss_dict = loss_fn(action_logits, turn_penalties, dummy_target_actions)
    
    print(f"\n=== Loss Components ===")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.4f}")
    
    print(f"\n=== Network Ready for Training ===")
    print("Actions: 0=Forward, 1=Back, 2=Left, 3=Right, 4=Up, 5=Down")
    print("The network can now be trained on your pathfinding datasets!")