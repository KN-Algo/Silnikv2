"""
Test script for the NNUE network with actual data from chess_data.npz
"""
import torch
from nnue import NNUE, load_dataset, prepare_batch


def test_with_real_data():
    """Test the NNUE network with real data from the dataset."""
    print("Loading dataset from NNUE/data/final/chess_data.npz...")
    
    # Load the dataset
    positions = load_dataset('NNUE/data/final/chess_data.npz')
    print(f"Loaded {len(positions)} positions")
    
    # Create the network
    nnue_net = NNUE(feature_count=4476, hidden_size=512)
    print("NNUE network created successfully")
    
    # Prepare a small batch from the dataset
    batch_size = 8
    white_indices, black_indices, labels = prepare_batch(positions, 0, batch_size)
    
    print(f"Batch prepared - White shape: {white_indices.shape}, Black shape: {black_indices.shape}")
    print(f"Labels: {labels}")
    
    # Run forward pass
    with torch.no_grad():  # No gradient computation needed for testing
        outputs = nnue_net(white_indices, black_indices)
        print(f"Network outputs: {outputs}")
        print(f"Output shape: {outputs.shape}")
        
        # Calculate mean squared error with labels
        mse = torch.mean((outputs - labels) ** 2)
        print(f"Mean Squared Error: {mse.item()}")
    
    print("Test completed successfully!")


def test_gradient_flow():
    """Test that gradients flow properly through the network."""
    print("\nTesting gradient flow...")
    
    # Create the network
    nnue_net = NNUE(feature_count=4476, hidden_size=512)
    
    # Create random inputs that match the data format
    batch_size = 4
    num_features = 30
    
    white_indices = torch.randint(0, 4476, (batch_size, num_features), dtype=torch.long)
    black_indices = torch.randint(0, 4476, (batch_size, num_features), dtype=torch.long)
    target = torch.randn(batch_size)  # Random targets for testing
    
    # Forward pass
    outputs = nnue_net(white_indices, black_indices)
    
    # Calculate loss
    loss = torch.mean((outputs - target) ** 2)
    print(f"Loss value: {loss.item()}")
    
    # Backward pass
    loss.backward()
    print("Gradient computation successful!")
    
    # Check if parameters have gradients
    for name, param in nnue_net.named_parameters():
        if param.grad is not None:
            print(f"Parameter {name} has gradient with shape: {param.grad.shape}")
        else:
            print(f"Parameter {name} has NO gradient")


if __name__ == "__main__":
    test_with_real_data()
    test_gradient_flow()