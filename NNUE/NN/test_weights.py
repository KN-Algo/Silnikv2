"""
Script to verify that saved weights can be loaded back into the model.
"""
import torch
import numpy as np
from nnue import NNUE
from train_nnue import save_weights_for_cpp, load_weights_from_cpp


def test_weight_loading():
    """Test that weights can be saved and loaded correctly."""
    print("Testing weight saving and loading...")
    
    # Create original network
    original_net = NNUE(feature_count=40050, hidden_size=512)  # Same as training script
    
    # Save the network weights
    print("Saving network weights...")
    save_weights_for_cpp(original_net, "test_weights.npz")
    
    # Create a new network and load the weights
    print("Creating new network and loading weights...")
    new_net = NNUE(feature_count=40050, hidden_size=512)
    loaded_net = load_weights_from_cpp(new_net, "test_weights.npz")
    
    # Compare a few parameters to make sure they match
    orig_params = dict(original_net.named_parameters())
    loaded_params = dict(loaded_net.named_parameters())
    
    all_match = True
    for name in orig_params:
        if name in loaded_params:
            if not torch.allclose(orig_params[name], loaded_params[name], atol=1e-6):
                print(f"Parameter {name} does not match after loading!")
                all_match = False
    
    if all_match:
        print("All parameters match after loading - weight saving/loading works correctly!")
    else:
        print("Some parameters do not match - there may be an issue with saving/loading.")
    
    # Clean up test file
    import os
    os.remove("test_weights.npz")


if __name__ == "__main__":
    test_weight_loading()