// Board.cpp
#include "chess/board/Board.h"
#include <iostream>
#include <sstream>
#include <string>
#include <cctype>


bool Board::loadFEN(const std::string& fen) {
    std::istringstream ss(fen);
    std::string placement, active, castl, ep;
    int half = 0, full = 1;

    if (!(ss >> placement >> active >> castl >> ep >> half >> full)) {
        return false;
    }

    // wyczyść planszę
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            board[r][c] = 0;

    // rozłóż bierki z pierwszego pola FEN
    int row = 0, col = 0;
    for (char ch : placement) {
        if (ch == '/') { ++row; col = 0; continue; }
        if (std::isdigit(static_cast<unsigned char>(ch))) { col += ch - '0'; continue; }
        if (row < 8 && col < 8) board[row][col++] = ch;
        else return false;
    }
    if (row != 7) return false;

    // pozostałe pola stanu
    activeColor    = (active == "w" ? 'w' : 'b');
    castling       = (castl == "-" ? "" : castl);
    enPassant      = (ep == "-" ? "" : ep);
    halfmoveClock  = half;
    fullmoveNumber = full;
    return true;
}


void Board::startBoard() {
    std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    setPositionFromFEN(fen);
}

void Board::setPositionFromFEN(const std::string& fen) {
    // Wyczyść planszę
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            board[r][c] = 0;
        }
    }
  
    int row = 0, col = 0;
    size_t i = 0;

    // Parsuj pozycję figur
    while (i < fen.length() && fen[i] != ' ') {
        char c = fen[i];
        if (c == '/') { row++; col = 0; }
        else if (isdigit(c)) { col += c - '0'; }
        else { board[row][col++] = c; }
        i++;
    }

    if (i < fen.length()) {
        // Parsuj stronę do ruchu
        activeColor = fen[++i];
        i += 2;

        // Parsuj roszady
        if (i < fen.length()) {
            size_t spacePos = fen.find(' ', i);
            if (spacePos != std::string::npos) {
                castling = fen.substr(i, spacePos - i);
                i = spacePos + 1;
            } else {
                castling = "-";
                i = fen.length();
            }
        }

        // Parsuj en passant
        if (i < fen.length()) {
            size_t spacePos = fen.find(' ', i);
            if (spacePos != std::string::npos) {
                enPassant = fen.substr(i, spacePos - i);
                i = spacePos + 1;
            } else {
                enPassant = "-";
                i = fen.length();
            }
        }

        // Parsuj licznik półruchów
        if (i < fen.length()) {
            size_t spacePos = fen.find(' ', i);
            if (spacePos != std::string::npos) {
                halfmoveClock = std::stoi(fen.substr(i, spacePos - i));
                i = spacePos + 1;
            } else {
                halfmoveClock = 0;
                i = fen.length();
            }
        }

        // Parsuj numer ruchu
        if (i < fen.length()) {
            fullmoveNumber = std::stoi(fen.substr(i));
        } else {
            fullmoveNumber = 1;
        }
    }

    gameStateManager.clearHistory();
    gameStateManager.addPosition(board, activeColor, castling, enPassant);
}

void Board::printBoard() const {
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            std::cout << (board[row][col] ? board[row][col] : '.') << ' ';
        }
        std::cout << '\n';
    }
    std::cout << "Active color: " << activeColor << '\n';
    std::cout << "Castling: " << castling << '\n';
    std::cout << "En passant: " << enPassant << '\n';
    std::cout << "Halfmove clock: " << halfmoveClock << '\n';
    std::cout << "Fullmove number: " << fullmoveNumber << '\n';
}

std::vector<Move> Board::getLegalMoves() const {
    return MoveGenerator::generateLegalMoves(board, activeColor, castling, enPassant);
}
bool Board::hasLegalMoves() const {
    return MoveGenerator::hasLegalMoves(board, activeColor, castling, enPassant);
}
bool Board::isInCheck() const {
    return MoveGenerator::isInCheck(board, activeColor);
}
GameState Board::getGameState() const {
    return gameStateManager.checkGameState(board, activeColor, castling, enPassant,
                                          halfmoveClock, hasLegalMoves(), isInCheck());
}
std::string Board::getGameStateString() const {
    return gameStateManager.getGameStateString(getGameState(), activeColor);
}

