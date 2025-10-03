#include "chess/rules/MoveGenerator.h"
#include <cctype>
#include <cmath>
#include "chess/rules/Attack.h"


std::vector<Move> MoveGenerator::generateAllMoves(const char board[8][8], char activeColor, 
                                                 const std::string& castling, const std::string& enPassant) {
    std::vector<Move> moves;
    
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            char piece = board[row][col];
            if (!piece) continue;
            
            bool whitePiece = std::isupper(piece);
            if ((activeColor == 'w' && !whitePiece) || (activeColor == 'b' && whitePiece))
                continue;
            
            switch (std::toupper(piece)) {
                case 'P':
                    generatePawnMoves(board, row, col, activeColor, enPassant, moves);
                    break;
                case 'N':
                    generateKnightMoves(board, row, col, activeColor, moves);
                    break;
                case 'B':
                    generateBishopMoves(board, row, col, activeColor, moves);
                    break;
                case 'R':
                    generateRookMoves(board, row, col, activeColor, moves);
                    break;
                case 'Q':
                    generateQueenMoves(board, row, col, activeColor, moves);
                    break;
                case 'K':
                    generateKingMoves(board, row, col, activeColor, castling, moves);
                    break;
            }
        }
    }
    
    return moves;
}

std::vector<Move> MoveGenerator::generateLegalMoves(const char board[8][8], char activeColor, 
                                                   const std::string& castling, const std::string& enPassant) {
    std::vector<Move> allMoves = generateAllMoves(board, activeColor, castling, enPassant);
    std::vector<Move> legalMoves;
    
    for (const Move& move : allMoves) {
        if (!wouldKingBeInCheck(board, move, activeColor)) {
            legalMoves.push_back(move);
        }
    }
    
    return legalMoves;
}

bool MoveGenerator::hasLegalMoves(const char board[8][8], char activeColor, 
                                 const std::string& castling, const std::string& enPassant) {
    std::vector<Move> allMoves = generateAllMoves(board, activeColor, castling, enPassant);
    
    for (const Move& move : allMoves) {
        if (!wouldKingBeInCheck(board, move, activeColor)) {
            return true;
        }
    }
    
    return false;
}

bool MoveGenerator::isInCheck(const char board[8][8], char activeColor) {
    char kingChar = (activeColor == 'w') ? 'K' : 'k';
    int kingRow = -1, kingCol = -1;
    
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (board[r][c] == kingChar) {
                kingRow = r;
                kingCol = c;
                break;
            }
        }
        if (kingRow != -1) break;
    }
    
    if (kingRow == -1) return false;
    
    char enemyColor = (activeColor == 'w') ? 'b' : 'w';
    return Attack::isSquareAttacked(board, kingRow, kingCol, enemyColor);
}

bool MoveGenerator::wouldKingBeInCheck(const char board[8][8], const Move& move, char activeColor) {
    // skopiuj planszę
    char tmp[8][8];
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            tmp[r][c] = board[r][c];

    char piece = board[move.fromRow][move.fromCol];
    bool isPawn  = std::toupper(piece) == 'P';
    bool isKing  = std::toupper(piece) == 'K';

    // --- 1) En passant: usuń pionka bitego z pola "obok" (na rzędzie startowym bijącego) ---
    bool epCapture = isPawn
                  && move.fromCol != move.toCol           // ruch po skosie
                  && board[move.toRow][move.toCol] == 0;  // pole docelowe puste => to musi być EP
    if (epCapture) {
        int capRow = move.fromRow;        // pion bity stoi na rzędzie bijącego
        int capCol = move.toCol;          // w kolumnie pola docelowego
        tmp[capRow][capCol] = 0;          // zdejmij pion bity EP
    }

    // --- 2) Roszada: przestaw też wieżę w symulacji (żeby nie fałszować ataków po linii) ---
    bool castling = isKing && std::abs(move.toCol - move.fromCol) == 2 && move.toRow == move.fromRow;
    if (castling) {
        int r = move.fromRow;
        if (move.toCol == 6) {          // O-O
            tmp[r][5] = tmp[r][7];
            tmp[r][7] = 0;
        } else if (move.toCol == 2) {   // O-O-O
            tmp[r][3] = tmp[r][0];
            tmp[r][0] = 0;
        }
    }

    // --- 3) Wykonaj ruch figury na planszy tymczasowej ---
    tmp[move.toRow][move.toCol]   = tmp[move.fromRow][move.fromCol];
    tmp[move.fromRow][move.fromCol] = 0;

    // --- 4) Sprawdź, czy nasz król jest w szachu po ruchu ---
    return isInCheck(tmp, activeColor);
}





void MoveGenerator::generatePawnMoves(const char board[8][8], int row, int col, char activeColor, 
                                     const std::string& enPassant, std::vector<Move>& moves) {
    char piece = board[row][col];
    bool whitePiece = std::isupper(piece);
    int dir = whitePiece ? -1 : 1;
    int startRow = whitePiece ? 6 : 1;
    
    if (row + dir >= 0 && row + dir < 8 && board[row + dir][col] == 0) {
        // Sprawdź czy to promocja
        if ((whitePiece && row + dir == 0) || (!whitePiece && row + dir == 7)) {
            // Generuj ruchy z promocją do wszystkich figur
            moves.push_back({row, col, row + dir, col, piece, 0, 'Q'}); // Hetman
            moves.push_back({row, col, row + dir, col, piece, 0, 'R'}); // Wieża
            moves.push_back({row, col, row + dir, col, piece, 0, 'B'}); // Goniec
            moves.push_back({row, col, row + dir, col, piece, 0, 'N'}); // Skoczek
        } else {
            moves.push_back({row, col, row + dir, col, piece, 0});
        }
        
        if (row == startRow && board[row + 2*dir][col] == 0) {
            moves.push_back({row, col, row + 2*dir, col, piece, 0});
        }
    }
    
    for (int dc = -1; dc <= 1; dc += 2) {
        if (col + dc >= 0 && col + dc < 8 && row + dir >= 0 && row + dir < 8) {
            char target = board[row + dir][col + dc];
            if (target && (std::isupper(target) != whitePiece)) {
                // Sprawdź czy to promocja
                if ((whitePiece && row + dir == 0) || (!whitePiece && row + dir == 7)) {
                    // Generuj ruchy z promocją do wszystkich figur
                    moves.push_back({row, col, row + dir, col + dc, piece, target, 'Q'}); // Hetman
                    moves.push_back({row, col, row + dir, col + dc, piece, target, 'R'}); // Wieża
                    moves.push_back({row, col, row + dir, col + dc, piece, target, 'B'}); // Goniec
                    moves.push_back({row, col, row + dir, col + dc, piece, target, 'N'}); // Skoczek
                } else {
                    moves.push_back({row, col, row + dir, col + dc, piece, target});
                }
            }
        }
    }
    
    if (enPassant != "-" && enPassant.length() >= 2) {
        int epCol = enPassant[0] - 'a';
        int epRow = '8' - enPassant[1];
        if (std::abs(col - epCol) == 1 && row + dir == epRow) {
            moves.push_back({row, col, epRow, epCol, piece, 0});
        }
    }
}

void MoveGenerator::generateKnightMoves(const char board[8][8], int row, int col, char activeColor, std::vector<Move>& moves) {
    char piece = board[row][col];
    bool whitePiece = std::isupper(piece);
    
    const int knightMoves[8][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},
                                   {1,-2},{1,2},{2,-1},{2,1}};
    
    for (auto& m : knightMoves) {
        int newRow = row + m[0];
        int newCol = col + m[1];
        
        if (newRow >= 0 && newRow < 8 && newCol >= 0 && newCol < 8) {
            char target = board[newRow][newCol];
            if (!target || (std::isupper(target) != whitePiece)) {
                moves.push_back({row, col, newRow, newCol, piece, target});
            }
        }
    }
}

void MoveGenerator::generateBishopMoves(const char board[8][8], int row, int col, char activeColor, std::vector<Move>& moves) {
    char piece = board[row][col];
    bool whitePiece = std::isupper(piece);
    
    const int directions[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    
    for (auto& dir : directions) {
        for (int i = 1; i < 8; i++) {
            int newRow = row + dir[0] * i;
            int newCol = col + dir[1] * i;
            
            if (newRow < 0 || newRow >= 8 || newCol < 0 || newCol >= 8) break;
            
            char target = board[newRow][newCol];
            if (!target) {
                moves.push_back({row, col, newRow, newCol, piece, 0});
            } else {
                if (std::isupper(target) != whitePiece) {
                    moves.push_back({row, col, newRow, newCol, piece, target});
                }
                break;
            }
        }
    }
}

void MoveGenerator::generateRookMoves(const char board[8][8], int row, int col, char activeColor, std::vector<Move>& moves) {
    char piece = board[row][col];
    bool whitePiece = std::isupper(piece);
    
    const int directions[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    
    for (auto& dir : directions) {
        for (int i = 1; i < 8; i++) {
            int newRow = row + dir[0] * i;
            int newCol = col + dir[1] * i;
            
            if (newRow < 0 || newRow >= 8 || newCol < 0 || newCol >= 8) break;
            
            char target = board[newRow][newCol];
            if (!target) {
                moves.push_back({row, col, newRow, newCol, piece, 0});
            } else {
                if (std::isupper(target) != whitePiece) {
                    moves.push_back({row, col, newRow, newCol, piece, target});
                }
                break;
            }
        }
    }
}

void MoveGenerator::generateQueenMoves(const char board[8][8], int row, int col, char activeColor, std::vector<Move>& moves) {
    generateBishopMoves(board, row, col, activeColor, moves);
    generateRookMoves(board, row, col, activeColor, moves);
}

void MoveGenerator::generateKingMoves(const char board[8][8], int row, int col, char activeColor, 
                                     const std::string& castling, std::vector<Move>& moves) {
    char piece = board[row][col];
    bool whitePiece = std::isupper(piece);
    
    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue;
            
            int newRow = row + dr;
            int newCol = col + dc;
            
            if (newRow >= 0 && newRow < 8 && newCol >= 0 && newCol < 8) {
                char target = board[newRow][newCol];
                if (!target || (std::isupper(target) != whitePiece)) {
                    moves.push_back({row, col, newRow, newCol, piece, target});
                }
            }
        }
    }
    
    if (whitePiece && row == 7 && col == 4) {
        if (castling.find('K') != std::string::npos && 
            board[7][5] == 0 && board[7][6] == 0 && board[7][7] == 'R' &&
            !Attack::isSquareAttacked(board, 7, 4, 'b') &&
            !Attack::isSquareAttacked(board, 7, 5, 'b') &&
            !Attack::isSquareAttacked(board, 7, 6, 'b')) {
            moves.push_back({7, 4, 7, 6, 'K', 0});
        }
        if (castling.find('Q') != std::string::npos && 
            board[7][3] == 0 && board[7][2] == 0 && board[7][1] == 0 && board[7][0] == 'R' &&
            !Attack::isSquareAttacked(board, 7, 4, 'b') &&
            !Attack::isSquareAttacked(board, 7, 3, 'b') &&
            !Attack::isSquareAttacked(board, 7, 2, 'b')) {
            moves.push_back({7, 4, 7, 2, 'K', 0});
        }
    }
    
    if (!whitePiece && row == 0 && col == 4) {
        if (castling.find('k') != std::string::npos && 
            board[0][5] == 0 && board[0][6] == 0 && board[0][7] == 'r' &&
            !Attack::isSquareAttacked(board, 0, 4, 'w') &&
            !Attack::isSquareAttacked(board, 0, 5, 'w') &&
            !Attack::isSquareAttacked(board, 0, 6, 'w')) {
            moves.push_back({0, 4, 0, 6, 'k', 0});
        }
        if (castling.find('q') != std::string::npos && 
            board[0][3] == 0 && board[0][2] == 0 && board[0][1] == 0 && board[0][0] == 'r' &&
            !Attack::isSquareAttacked(board, 0, 4, 'w') &&
            !Attack::isSquareAttacked(board, 0, 3, 'w') &&
            !Attack::isSquareAttacked(board, 0, 2, 'w')) {
            moves.push_back({0, 4, 0, 2, 'k', 0});
        }
    }
}