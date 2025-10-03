// Board.h
#pragma once
#include <string>
#include <vector>
#include "chess/rules/MoveGenerator.h"
#include "chess/game/GameState.h"

struct Move; // z move.h

class Board {
public:
    // Dane pozycji
    char board[8][8]{};      // 0 = puste, inaczej litera figury
    char activeColor{'w'};   // 'w' / 'b'
    std::string castling{"KQkq"};
    std::string enPassant{"-"};  // np. "e3" lub "-"
    int halfmoveClock{0};
    int fullmoveNumber{1};

    GameStateManager gameStateManager; // możesz dać do private, jeśli wolisz

    std::vector<Move> getLegalMoves() const;
    bool hasLegalMoves() const;
    bool isInCheck() const;
    GameState getGameState() const;
    std::string getGameStateString() const;
    bool loadFEN(const std::string& fen);

    // Interfejs publiczny
    void startBoard();           // inicjalizacja z FEN
    void setPositionFromFEN(const std::string& fen);  // ustaw pozycję z FEN
    void printBoard() const;

    // Logika — dalej implementacje w osobnych .cpp
    bool isPathClear(int r1, int c1, int r2, int c2) const;
    bool isSquareAttacked(int row, int col, char byColor) const;
    char getPieceAt(int row, int col) const { return board[row][col]; }

    std::string checkCastlingReason(const Move& move) const;

    bool isMoveValid(const Move& move);
    void makeMove(const Move& move);
};
