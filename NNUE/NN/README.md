# NNUE Training and Usage Guide

This project implements an NNUE (Neural Network Unified Evaluation) network for chess evaluation, with training capabilities and C++ integration support.

## Files

- `nnue.py`: Implementation of the NNUE network architecture
- `train_nnue.py`: Training script with automatic feature size detection
- `nnue_weights.npz`: Compressed numpy file containing trained weights (saved after training)
- `nnue_weights.h`: C++ header file with trained weights in C++ format
- `nnue_cpp_example.cpp`: Example C++ implementation showing how to use the trained weights

## Training the Network

To train the network:

```bash
python3 NNUE/NN/train_nnue.py
```

The training script will:
1. Automatically detect the maximum feature index in your dataset
2. Create an appropriately sized network
3. Train for the specified number of epochs
4. Save weights in both `.npz` and C++ compatible `.h` formats

## Using Trained Weights in C++

The `nnue_weights.h` file contains all network weights in C++ format as `std::array<float, N>` variables. These can be directly included in your C++ chess engine.

The `nnue_cpp_example.cpp` file shows how to:
1. Load the weights into C++ arrays
2. Implement the forward pass of your trained network
3. Evaluate chess positions

## Network Architecture

The implemented NNUE network has:
- Input layer: Sparse feature indices for white and black pieces
- Feature transformer: Linear layer mapping to hidden representation (512 units)
- Hidden layers: 3 layers with [512, 256, 128] units and ReLU activation
- Output: Single value representing position evaluation

## Data Format

The network expects training data in the format from `NNUE/data/final/chess_data.npz`:
- Each position contains `white_features` and `black_features` (arrays of feature indices)
- Features are padded/truncated to a fixed length (default 30)
- Labels represent the evaluation score for the position

## Feature Indexing

The network automatically detects the maximum feature index in the dataset and creates the appropriate input size. Current implementation supports up to feature index 40049.

## Integration with C++ Engine

To use the trained weights in a C++ chess engine:

1. Include `nnue_weights.h` in your project
2. Implement the forward pass logic (similar to the example in `nnue_cpp_example.cpp`)
3. Feed the appropriate feature indices for your current position
4. Use the returned evaluation score for position assessment

The weights are stored as `constexpr` arrays for efficient compilation and use in performance-critical applications.