/**
 * Example C++ implementation showing how to use the trained NNUE weights
 * for inference in a chess engine.
 */
#include "nnue_weights.h"
#include <vector>
#include <algorithm>
#include <cstring>

class NNUEEvaluator {
private:
    // Configuration constants
    static constexpr int INPUT_SIZE = 40050;  // Maximum feature index + 1
    static constexpr int HIDDEN_SIZE = 512;
    
    // Feature transformer weights and biases
    float featureWeights[HIDDEN_SIZE][INPUT_SIZE];
    float hiddenWeights[3][HIDDEN_SIZE * 2][HIDDEN_SIZE];  // 3 hidden layers
    float hiddenBiases[3][HIDDEN_SIZE];  // 3 hidden layers
    float outputWeight[HIDDEN_SIZE / 4];  // Final layer
    float outputBias;
    
    // Activation function
    float relu(float x) {
        return x > 0 ? x : 0;
    }
    
    // Clamp function
    float clamp(float x, float min_val, float max_val) {
        return std::min(std::max(x, min_val), max_val);
    }

public:
    NNUEEvaluator() {
        // Initialize weights from the generated arrays
        // Feature transformer
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                featureWeights[i][j] = feature_transformer_weight_[i * INPUT_SIZE + j];
            }
        }
        
        // Hidden layers
        // First hidden layer (input: HIDDEN_SIZE * 2, output: HIDDEN_SIZE)
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            for (int j = 0; j < HIDDEN_SIZE * 2; ++j) {
                hiddenWeights[0][i][j] = hidden_layers_0_weight_[i * HIDDEN_SIZE * 2 + j];
            }
            hiddenBiases[0][i] = hidden_layers_0_bias_[i];
        }
        
        // Second hidden layer (input: HIDDEN_SIZE, output: HIDDEN_SIZE / 2)
        for (int i = 0; i < HIDDEN_SIZE / 2; ++i) {
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                hiddenWeights[1][i][j] = hidden_layers_2_weight_[i * HIDDEN_SIZE + j];
            }
            hiddenBiases[1][i] = hidden_layers_2_bias_[i];
        }
        
        // Third hidden layer (input: HIDDEN_SIZE / 2, output: HIDDEN_SIZE / 4)
        for (int i = 0; i < HIDDEN_SIZE / 4; ++i) {
            for (int j = 0; j < HIDDEN_SIZE / 2; ++j) {
                hiddenWeights[2][i][j] = hidden_layers_4_weight_[i * HIDDEN_SIZE / 2 + j];
            }
            hiddenBiases[2][i] = hidden_layers_4_bias_[i];
        }
        
        // Output layer
        for (int i = 0; i < HIDDEN_SIZE / 4; ++i) {
            outputWeight[i] = output_layer_weight_[i];
        }
        outputBias = output_layer_bias_[0];
    }
    
    /**
     * Evaluate a chess position using the trained NNUE network.
     * 
     * @param whiteFeatures: Array of feature indices for white pieces (padded to fixed size)
     * @param whiteFeatureCount: Number of active white features
     * @param blackFeatures: Array of feature indices for black pieces (padded to fixed size)
     * @param blackFeatureCount: Number of active black features
     * @return Evaluation score for the position
     */
    float evaluate(const std::vector<int>& whiteFeatures, 
                   const std::vector<int>& blackFeatures) {
        
        // Initialize input vectors
        std::vector<float> whiteInput(INPUT_SIZE, 0.0f);
        std::vector<float> blackInput(INPUT_SIZE, 0.0f);
        
        // Set active features to 1.0
        for (int feature : whiteFeatures) {
            if (feature >= 0 && feature < INPUT_SIZE) {
                whiteInput[feature] = 1.0f;
            }
        }
        
        for (int feature : blackFeatures) {
            if (feature >= 0 && feature < INPUT_SIZE) {
                blackInput[feature] = 1.0f;
            }
        }
        
        // Transform features using the feature transformer
        std::vector<float> whiteTransformed(HIDDEN_SIZE, 0.0f);
        std::vector<float> blackTransformed(HIDDEN_SIZE, 0.0f);
        
        // Apply transformation for white features
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < INPUT_SIZE; ++j) {
                sum += whiteInput[j] * featureWeights[i][j];
            }
            whiteTransformed[i] = clamp(sum, -1.0f, 1.0f);
        }
        
        // Apply transformation for black features
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < INPUT_SIZE; ++j) {
                sum += blackInput[j] * featureWeights[i][j];
            }
            blackTransformed[i] = clamp(sum, -1.0f, 1.0f);
        }
        
        // Combine white and black transformed features
        std::vector<float> combined(HIDDEN_SIZE * 2);
        std::copy(whiteTransformed.begin(), whiteTransformed.end(), combined.begin());
        std::copy(blackTransformed.begin(), blackTransformed.end(), combined.begin() + HIDDEN_SIZE);
        
        // Forward pass through hidden layers
        std::vector<float> current = combined;
        
        for (int layer = 0; layer < 3; ++layer) {
            int nextSize = (layer == 0) ? HIDDEN_SIZE : 
                          (layer == 1) ? HIDDEN_SIZE / 2 : HIDDEN_SIZE / 4;
            std::vector<float> next(nextSize, 0.0f);
            
            for (int i = 0; i < nextSize; ++i) {
                float sum = hiddenBiases[layer][i];  // Start with bias
                for (int j = 0; j < current.size(); ++j) {
                    sum += current[j] * hiddenWeights[layer][i][j];
                }
                next[i] = relu(sum);  // Apply ReLU activation
            }
            
            current = next;
        }
        
        // Final output
        float output = outputBias;
        for (int i = 0; i < current.size(); ++i) {
            output += current[i] * outputWeight[i];
        }
        
        return output;
    }
};

// Example usage
#include <iostream>

int main() {
    std::cout << "NNUE Evaluator Example" << std::endl;
    
    // Initialize the evaluator with trained weights
    NNUEEvaluator evaluator;
    
    // Example: evaluate a simple position
    // In a real chess engine, these would come from the board position
    std::vector<int> whiteFeatures = {0, 1, 2, 3};  // Example feature indices
    std::vector<int> blackFeatures = {4, 5, 6, 7};  // Example feature indices
    
    float score = evaluator.evaluate(whiteFeatures, blackFeatures);
    
    std::cout << "Evaluation score: " << score << std::endl;
    
    return 0;
}