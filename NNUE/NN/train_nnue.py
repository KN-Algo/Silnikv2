"""
Training script for the NNUE network with automatic feature size detection.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from nnue import NNUE, load_dataset, prepare_batch


class ChessDataset(Dataset):
    """Dataset wrapper for chess positions."""
    
    def __init__(self, positions, max_features=30, max_feature_idx=None):
        self.positions = positions
        self.max_features = max_features
        self.max_feature_idx = max_feature_idx
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        pos = self.positions[idx]
        
        # Pad or truncate feature lists to max_features length
        white_feat = pos['white_features'][:self.max_features]
        black_feat = pos['black_features'][:self.max_features]
        
        # Pad with 0s if necessary
        if len(white_feat) < self.max_features:
            white_feat.extend([0] * (self.max_features - len(white_feat)))
        if len(black_feat) < self.max_features:
            black_feat.extend([0] * (self.max_features - len(black_feat)))
        
        return {
            'white_indices': torch.tensor(white_feat, dtype=torch.long),
            'black_indices': torch.tensor(black_feat, dtype=torch.long),
            'label': torch.tensor(pos['label'], dtype=torch.float)
        }


def find_max_feature_index(positions, sample_size=10000):
    """
    Find the maximum feature index in the dataset.
    
    Args:
        positions: List of position dictionaries
        sample_size: Number of positions to sample
    
    Returns:
        Maximum feature index found
    """
    max_idx = 0
    sample_positions = positions[:sample_size] if len(positions) > sample_size else positions
    
    print(f"Analyzing {len(sample_positions)} positions to find max feature index...")
    for i, pos in enumerate(sample_positions):
        if i % 1000 == 0:
            print(f"Processed {i} positions...")
        
        max_white = max(pos['white_features']) if pos['white_features'] else 0
        max_black = max(pos['black_features']) if pos['black_features'] else 0
        max_idx = max(max_idx, max_white, max_black)
    
    print(f"Maximum feature index found: {max_idx}")
    return max_idx + 1  # +1 to account for 0-indexing


def train_nnue(network, positions, epochs=10, batch_size=1024, learning_rate=0.001, validation_split=0.1):
    """
    Train the NNUE network.
    
    Args:
        network: The NNUE network to train
        positions: List of chess positions
        epochs: Number of training epochs
        batch_size: Size of training batches
        learning_rate: Learning rate for optimizer
        validation_split: Fraction of data to use for validation
    """
    # Split data into training and validation sets
    val_size = int(len(positions) * validation_split)
    train_positions = positions[val_size:]
    val_positions = positions[:val_size]
    
    print(f"Training on {len(train_positions)} positions, validating on {len(val_positions)} positions")
    
    # Create datasets and data loaders
    train_dataset = ChessDataset(train_positions)
    val_dataset = ChessDataset(val_positions)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        network.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            white_indices = batch['white_indices']
            black_indices = batch['black_indices']
            labels = batch['label']
            
            optimizer.zero_grad()
            outputs = network(white_indices, black_indices)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:  # Print progress every 50 batches
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.6f}')
        
        # Validation phase
        network.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                white_indices = batch['white_indices']
                black_indices = batch['black_indices']
                labels = batch['label']
                
                outputs = network(white_indices, black_indices)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')


def save_weights_for_cpp(network, filename):
    """
    Save network weights in a format that can be loaded in C++.
    This creates a file with raw weight data that can be easily read in C++.
    """
    network.eval()  # Set to evaluation mode
    
    # Dictionary to hold all parameters
    weights = {}
    
    # Extract all parameters
    for name, param in network.named_parameters():
        weights[name] = param.data.cpu().numpy()
    
    # Save using numpy
    np.savez_compressed(filename, **weights)
    print(f"Weights saved to {filename}")
    
    return weights


def load_weights_from_cpp(network, filename):
    """
    Load network weights from the format saved for C++.
    """
    weights = np.load(filename)
    
    # Load each parameter
    network_state_dict = {}
    for name, param in network.named_parameters():
        if name in weights:
            network_state_dict[name] = torch.from_numpy(weights[name])
    
    network.load_state_dict(network_state_dict)
    print(f"Weights loaded from {filename}")
    
    return network


def convert_weights_to_cpp_header(network, filename):
    """
    Convert the network weights to C++ header format.
    """
    with open(filename, 'w') as f:
        f.write("// Auto-generated C++ header file with NNUE weights\n")
        f.write("#ifndef NNUE_WEIGHTS_H\n")
        f.write("#define NNUE_WEIGHTS_H\n\n")
        f.write("#include <array>\n#include <cstddef>\n\n")
        
        # Extract and write all parameters
        for name, param in network.named_parameters():
            shape = param.shape
            total_elements = param.numel()
            
            # Generate C++ array type and name
            cpp_name = name.replace('.', '_').replace('weight', 'weight_').replace('bias', 'bias_')
            
            f.write(f"// {name}: shape {list(shape)}\n")
            f.write(f"constexpr std::array<float, {total_elements}> {cpp_name} = {{")
            
            # Write values
            values = param.data.cpu().numpy().flatten()
            for i, val in enumerate(values):
                if i % 10 == 0:
                    f.write("\n  ")
                f.write(f"{val:.10f}")
                if i < total_elements - 1:
                    f.write(", ")
            
            f.write("\n};\n\n")
        
        f.write("#endif // NNUE_WEIGHTS_H\n")
    
    print(f"C++ header file saved to {filename}")


if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    positions = load_dataset('NNUE/data/final/chess_data.npz')
    print(f"Loaded {len(positions)} positions")
    
    # Find the maximum feature index in the dataset
    max_feature_idx = find_max_feature_index(positions, sample_size=50000)  # Sample 50k positions
    
    # Create the network with the correct feature count
    print(f"Creating network with {max_feature_idx} features and 512 hidden units...")
    nnue_net = NNUE(feature_count=max_feature_idx, hidden_size=512)
    
    # Train the network (using a smaller subset for faster training)
    print("Starting training...")
    train_nnue(nnue_net, positions[:50000], epochs=3, batch_size=256)  # Using subset for faster training
    
    # Save the trained weights
    save_weights_for_cpp(nnue_net, "nnue_weights.npz")
    
    # Convert to C++ header format
    convert_weights_to_cpp_header(nnue_net, "nnue_weights.h")
    
    print("Training completed and weights saved!")