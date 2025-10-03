#pragma once
#include <string>
#include <unordered_map>

enum class GameState {
    PLAYING,
    CHECKMATE,
    STALEMATE,
    DRAW_50_MOVES,
    DRAW_REPETITION,
    DRAW_INSUFFICIENT_MATERIAL
};

class GameStateManager {
private:
    std::unordered_map<std::string, int> positionHistory;
    
    bool hasInsufficientMaterial(const char board[8][8]) const;
    std::string boardToString(const char board[8][8], char activeColor, 
                             const std::string& castling, const std::string& enPassant) const;

public:
    void addPosition(const char board[8][8], char activeColor, 
                    const std::string& castling, const std::string& enPassant);
    void clearHistory();
    
    GameState checkGameState(const char board[8][8], char activeColor, 
                           const std::string& castling, const std::string& enPassant,
                           int halfmoveClock, bool hasLegalMoves, bool isInCheck) const;
    std::string getGameStateString(GameState state, char activeColor) const;
};