
#include "chess/game/GameState.h"
#include <cctype>

void GameStateManager::addPosition(const char board[8][8], char activeColor, 
                                  const std::string& castling, const std::string& enPassant) {
    std::string position = boardToString(board, activeColor, castling, enPassant);
    positionHistory[position]++;
}

void GameStateManager::clearHistory() {
    positionHistory.clear();
}

GameState GameStateManager::checkGameState(const char board[8][8], char activeColor, 
                                         const std::string& castling, const std::string& enPassant,
                                         int halfmoveClock, bool hasLegalMoves, bool isInCheck) const {
    if (halfmoveClock >= 100) {
        return GameState::DRAW_50_MOVES;
    }
    
    std::string currentPosition = boardToString(board, activeColor, castling, enPassant);
    auto it = positionHistory.find(currentPosition);
    if (it != positionHistory.end() && it->second >= 3) {
        return GameState::DRAW_REPETITION;
    }
    
    if (hasInsufficientMaterial(board)) {
        return GameState::DRAW_INSUFFICIENT_MATERIAL;
    }
    
    if (!hasLegalMoves) {
        if (isInCheck) {
            return GameState::CHECKMATE;
        } else {
            return GameState::STALEMATE;
        }
    }
    
    return GameState::PLAYING;
}

std::string GameStateManager::getGameStateString(GameState state, char activeColor) const {
    switch (state) {
        case GameState::PLAYING:
            return "Game in progress";
        case GameState::CHECKMATE:
            return (activeColor == 'w') ? "Czarny wygrywa przez mata!" : "Biały wygrywa przez mata!";
        case GameState::STALEMATE:
            return "Remis przez pata!";
        case GameState::DRAW_50_MOVES:
            return "Remis przez zasadę 50 ruchów!";
        case GameState::DRAW_REPETITION:
            return "Remis przez trzykrotne powtórzenie pozycji!";
        case GameState::DRAW_INSUFFICIENT_MATERIAL:
            return "Remis przez niewystarczający materiał!";
        default:
            return "Unknown state";
    }
}

std::string GameStateManager::boardToString(const char board[8][8], char activeColor, 
                                           const std::string& castling, const std::string& enPassant) const {
    std::string result;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            result += (board[r][c] ? board[r][c] : '.');
        }
    }
    result += activeColor;
    result += castling;
    result += enPassant;
    return result;
}

bool GameStateManager::hasInsufficientMaterial(const char board[8][8]) const {
    int wKing = 0, bKing = 0;
    int wQueen = 0, bQueen = 0;
    int wRook = 0, bRook = 0;
    int wBishop = 0, bBishop = 0;
    int wKnight = 0, bKnight = 0;
    int wPawn = 0, bPawn = 0;
    
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            switch (board[r][c]) {
                case 'K': wKing++; break;
                case 'k': bKing++; break;
                case 'Q': wQueen++; break;
                case 'q': bQueen++; break;
                case 'R': wRook++; break;
                case 'r': bRook++; break;
                case 'B': wBishop++; break;
                case 'b': bBishop++; break;
                case 'N': wKnight++; break;
                case 'n': bKnight++; break;
                case 'P': wPawn++; break;
                case 'p': bPawn++; break;
            }
        }
    }
    
    if (wPawn || bPawn || wQueen || bQueen || wRook || bRook) {
        return false;
    }
    
    if (wBishop == 0 && bBishop == 0 && wKnight == 0 && bKnight == 0) {
        return true;
    }
    
    if ((wBishop == 1 && bBishop == 0 && wKnight == 0 && bKnight == 0) ||
        (wBishop == 0 && bBishop == 1 && wKnight == 0 && bKnight == 0)) {
        return true;
    }
    
    if ((wKnight == 1 && bKnight == 0 && wBishop == 0 && bBishop == 0) ||
        (wKnight == 0 && bKnight == 1 && wBishop == 0 && bBishop == 0)) {
        return true;
    }
    
    return false;
}