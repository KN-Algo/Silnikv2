"""
Example of how to perform inference with the trained NNUE model.
"""
import torch
import numpy as np
from nnue import NNUE, load_dataset, prepare_batch


def run_inference_example():
    """Example of how to run inference with a trained model."""
    print("Running inference example...")
    
    # Create a new network with the same architecture used in training
    network = NNUE(feature_count=40050, hidden_size=512)
    
    # Load the trained weights (if available)
    try:
        weights = np.load('nnue_weights.npz')
        network_state_dict = {}
        for name, param in network.named_parameters():
            if name in weights:
                network_state_dict[name] = torch.from_numpy(weights[name])
        network.load_state_dict(network_state_dict)
        print("Loaded trained weights successfully!")
    except FileNotFoundError:
        print("Trained weights file not found, using randomly initialized network")
    
    # Set network to evaluation mode
    network.eval()
    
    # Example 1: Random features (simulating a position)
    print("\nExample 1: Random features")
    batch_size = 2
    max_features = 30
    
    white_indices = torch.randint(0, 40050, (batch_size, max_features))
    black_indices = torch.randint(0, 40050, (batch_size, max_features))
    
    with torch.no_grad():
        outputs = network(white_indices, black_indices)
        print(f"Input shapes - White: {white_indices.shape}, Black: {black_indices.shape}")
        print(f"Output evaluations: {outputs}")
    
    # Example 2: Using actual data from the dataset (if available)
    try:
        print("\nExample 2: Using actual dataset positions")
        positions = load_dataset('NNUE/data/final/chess_data.npz')
        
        # Prepare a small batch
        white_indices, black_indices, labels = prepare_batch(positions, 0, 3, max_features=30)
        
        with torch.no_grad():
            outputs = network(white_indices, black_indices)
            print(f"Actual data shapes - White: {white_indices.shape}, Black: {black_indices.shape}")
            print(f"Labels (expected): {labels}")
            print(f"Network outputs (predictions): {outputs}")
            print(f"Absolute errors: {torch.abs(outputs - labels)}")
    except FileNotFoundError:
        print("\nDataset file not found, skipping actual data example.")
    
    print("\nInference example completed!")


if __name__ == "__main__":
    run_inference_example()