"""
NNUE (Neural Network Unified Evaluation) implementation for chess.
Based on the NNUE architecture used in Stockfish and other chess engines.
"""
import numpy as np
import torch
import torch.nn as nn
import os


class NNUE(nn.Module):
    """
    NNUE (Efficiently Updatable Neural Network) for chess evaluation.
    
    Input: Features for white and black pieces separately
    Processing: Feature transformation -> Hidden layers -> Output
    Output: Evaluation score for the position (from white's perspective)
    """
    
    def __init__(self, feature_count=4476, hidden_size=512):
        """
        Initialize the NNUE network.
        
        Args:
            feature_count: Number of possible input features (max feature index + 1)
            hidden_size: Size of the hidden layers
        """
        super(NNUE, self).__init__()
        
        # Feature transformer: converts sparse feature indices to dense feature vectors
        self.feature_transformer = nn.Linear(feature_count, hidden_size, bias=False)
        
        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 because we concatenate white and black
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU()
        )
        
        # Output layer (single value for evaluation)
        self.output_layer = nn.Linear(hidden_size // 4, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with small random values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, white_indices, black_indices):
        """
        Forward pass of the network.
        
        Args:
            white_indices: Tensor of shape (batch_size, num_features) with feature indices for white
            black_indices: Tensor of shape (batch_size, num_features) with feature indices for black
        
        Returns:
            output: Tensor of shape (batch_size, 1) with evaluation scores
        """
        batch_size = white_indices.size(0)
        
        # Create one-hot encoded feature vectors for white and black
        white_features = self._create_features(white_indices, batch_size)
        black_features = self._create_features(black_indices, batch_size)
        
        # Transform features using the feature transformer
        white_transformed = self.feature_transformer(white_features)
        black_transformed = self.feature_transformer(black_features)
        
        # Clamp the transformed features for stability (as in traditional NNUE)
        white_transformed = torch.clamp(white_transformed, -1, 1)
        black_transformed = torch.clamp(black_transformed, -1, 1)
        
        # Concatenate white and black features
        combined_features = torch.cat([white_transformed, black_transformed], dim=1)
        
        # Pass through hidden layers
        hidden = self.hidden_layers(combined_features)
        
        # Output evaluation
        output = self.output_layer(hidden)
        
        # Squeeze to get shape (batch_size,) instead of (batch_size, 1)
        return output.squeeze(-1)
    
    def _create_features(self, indices, batch_size):
        """
        Convert sparse feature indices to dense feature vectors.
        
        Args:
            indices: Tensor of shape (batch_size, num_features) with feature indices
            batch_size: Size of the batch
        
        Returns:
            features: Dense feature tensor of shape (batch_size, feature_count)
        """
        # Create a zero tensor with the correct shape
        device = indices.device
        feature_count = self.feature_transformer.in_features
        features = torch.zeros((batch_size, feature_count), device=device)
        
        # Create batch indices for advanced indexing
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(indices)
        
        # Set the appropriate feature indices to 1 (one-hot encoding)
        features[batch_idx, indices] = 1.0
        
        return features


def load_dataset(file_path):
    """
    Load chess positions dataset from .npz file.
    
    Args:
        file_path: Path to the .npz file containing chess positions
    
    Returns:
        List of position dictionaries
    """
    data = np.load(file_path, allow_pickle=True)
    positions = data['positions']
    return positions


def prepare_batch(positions, batch_start, batch_size, max_features=30):
    """
    Prepare a batch of positions for training/inference.
    
    Args:
        positions: List of position dictionaries
        batch_start: Starting index for the batch
        batch_size: Number of positions in the batch
        max_features: Maximum number of features (sequences will be padded/truncated)
    
    Returns:
        white_indices: Tensor of white feature indices
        black_indices: Tensor of black feature indices
        labels: Tensor of labels
    """
    actual_batch_size = min(batch_size, len(positions) - batch_start)
    
    white_features_batch = []
    black_features_batch = []
    labels_batch = []
    
    for i in range(actual_batch_size):
        pos_idx = batch_start + i
        pos = positions[pos_idx]
        
        # Pad or truncate feature lists to max_features length
        white_feat = pos['white_features'][:max_features]
        black_feat = pos['black_features'][:max_features]
        
        # Pad with 0s if necessary
        if len(white_feat) < max_features:
            white_feat.extend([0] * (max_features - len(white_feat)))
        if len(black_feat) < max_features:
            black_feat.extend([0] * (max_features - len(black_feat)))
        
        white_features_batch.append(white_feat)
        black_features_batch.append(black_feat)
        labels_batch.append(pos['label'])
    
    # Convert to tensors
    white_indices = torch.tensor(white_features_batch, dtype=torch.long)
    black_indices = torch.tensor(black_features_batch, dtype=torch.long)
    labels = torch.tensor(labels_batch, dtype=torch.float)
    
    return white_indices, black_indices, labels


if __name__ == "__main__":
    # Example usage
    print("NNUE network implementation for chess evaluation")
    print("Creating network with 4476 features and 512 hidden units...")
    
    # Create the network
    nnue_net = NNUE(feature_count=4476, hidden_size=512)
    
    # Example forward pass with random inputs
    batch_size = 4
    max_features = 30  # Maximum number of features
    
    white_indices = torch.randint(0, 4476, (batch_size, max_features))
    black_indices = torch.randint(0, 4476, (batch_size, max_features))
    
    print(f"Input shape - White: {white_indices.shape}, Black: {black_indices.shape}")
    
    output = nnue_net(white_indices, black_indices)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    # Test with actual data from the dataset if available
    try:
        print("\nTesting with actual data:")
        positions = load_dataset('NNUE/data/final/chess_data.npz')
        white_indices, black_indices, labels = prepare_batch(positions, 0, 4)
        print(f"Actual data shapes - White: {white_indices.shape}, Black: {black_indices.shape}, Labels: {labels.shape}")
        
        with torch.no_grad():
            actual_outputs = nnue_net(white_indices, black_indices)
            print(f"Actual data outputs: {actual_outputs}")
    except FileNotFoundError:
        print("Dataset file not found, skipping actual data test.")