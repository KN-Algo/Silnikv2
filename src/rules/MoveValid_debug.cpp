// MoveValid.cpp
#include "chess/rules/MoveValid.h"
#include "chess/rules/Attack.h"
#include "chess/rules/Castling.h"
#include "chess/utils/Notation.h"
#include <algorithm>
#include <cctype>
#include <iostream>

using namespace notation;

bool Board::isMoveValid(const Move &move)
{
    std::cout << "DEBUG: isMoveValid called for move from (" << move.fromRow << "," << move.fromCol 
              << ") to (" << move.toRow << "," << move.toCol << ")" << std::endl;

    int r1 = move.fromRow;
    int c1 = move.fromCol;
    int r2 = move.toRow;
    int c2 = move.toCol;

    char piece = board[r1][c1];
    std::cout << "DEBUG: piece at source (" << r1 << "," << c1 << ") = '" << piece << "'" << std::endl;
    if (!piece) {
        std::cout << "DEBUG: No piece at source, returning false" << std::endl;
        return false;
    }

    bool whitePiece = std::isupper(piece);
    std::cout << "DEBUG: whitePiece = " << (whitePiece ? "true" : "false") << std::endl;
    if ((activeColor == 'w' && !whitePiece) || (activeColor == 'b' && whitePiece)) {
        std::cout << "DEBUG: Wrong color, returning false" << std::endl;
        return false;
    }

    std::cout << "DEBUG: piece at target (" << r2 << "," << c2 << ") = '" << board[r2][c2] << "'" << std::endl;
    std::cout << "DEBUG: std::isupper(board[r2][c2]) = " << (board[r2][c2] ? std::to_string(std::isupper(board[r2][c2])) : "0") << std::endl;
    std::cout << "DEBUG: whitePiece = " << whitePiece << std::endl;
    std::cout << "DEBUG: board[r2][c2] = '" << board[r2][c2] << "'" << std::endl;
    std::cout << "DEBUG: (std::isupper(board[r2][c2]) == whitePiece) = " << ((std::isupper(board[r2][c2]) == whitePiece) ? "true" : "false") << std::endl;
    
    if (board[r2][c2] && (std::isupper(board[r2][c2]) == whitePiece)) {
        std::cout << "DEBUG: Trying to capture own piece, returning false" << std::endl;
        return false;
    }

    int dr = r2 - r1;
    int dc = c2 - c1;
    int adr = std::abs(dr), adc = std::abs(dc);
    std::cout << "DEBUG: dr=" << dr << ", dc=" << dc << ", adr=" << adr << ", adc=" << adc << std::endl;

    switch (std::toupper(piece))
    {
    case 'P':
{
    std::cout << "DEBUG: Pawn move logic" << std::endl;
    int dir = whitePiece ? -1 : 1; // białe idą w górę (r--), czarne w dół (r++)
    bool isEnPassantCapture = false;
    int epCapRow = -1, epCapCol = -1;

    if (dc == 0) {
        std::cout << "DEBUG: Forward move" << std::endl;
        // zwykły ruch do przodu o 1
        if (dr == dir && board[r2][c2] == 0) {
            std::cout << "DEBUG: Simple forward move, OK" << std::endl;
            // OK
        }
        // ruch o 2 z rzędu startowego (6 dla białych, 1 dla czarnych)
        else if (( (whitePiece && r1 == 6) || (!whitePiece && r1 == 1) ) &&
                 dr == 2 * dir &&
                 board[r1 + dir][c1] == 0 && board[r2][c2] == 0)
        {
            std::cout << "DEBUG: Double forward move, OK" << std::endl;
            // OK (EP target ustawimy w makeMove)
        }
        else {
            std::cout << "DEBUG: Invalid forward move, returning false" << std::endl;
            return false;
        }
    }
    else if (adr == 1 && dr == dir) {
        std::cout << "DEBUG: Diagonal move (capture)" << std::endl;
        // bicie po skosie o 1: zwykłe lub en passant
        if (board[r2][c2] != 0) {
            std::cout << "DEBUG: Regular capture" << std::endl;
            // zwykłe bicie – sprawdź czy bijemy przeciwnika
            if (std::isupper(board[r2][c2]) == whitePiece) {
                std::cout << "DEBUG: Trying to capture own piece in diagonal move, returning false" << std::endl;
                return false; // nie można bić swojego
            }
        } else {
            std::cout << "DEBUG: En passant capture attempt" << std::endl;
            // --- EP: bicie w przelocie, gdy cel jest pusty, ale = enPassant ---
            if (enPassant != "-") {
                int epRow, epCol;
                if (algToCoord(enPassant, epRow, epCol) && epRow == r2 && epCol == c2) {
                    // sprawdzamy, czy faktycznie stoi pion przeciwnika "za" polem celu
                    int capRow = r2 - dir; // pion do zbicia stoi na rzędzie, z którego przechodziliśmy
                    int capCol = c2;
                    char capPiece = board[capRow][capCol];
                    if (capPiece != 0 && std::toupper(capPiece) == 'P' &&
                        (std::isupper(capPiece) != whitePiece))
                    {
                        std::cout << "DEBUG: Valid en passant capture" << std::endl;
                        isEnPassantCapture = true;
                        epCapRow = capRow; epCapCol = capCol;
                    } else {
                        std::cout << "DEBUG: Invalid en passant capture, returning false" << std::endl;
                        return false;
                    }
                } else {
                    std::cout << "DEBUG: En passant target mismatch, returning false" << std::endl;
                    return false;
                }
            } else {
                std::cout << "DEBUG: No en passant target, returning false" << std::endl;
                return false;
            }
        }
    }
    else {
        std::cout << "DEBUG: Invalid pawn move pattern, returning false" << std::endl;
        return false;
    }

    // --- wymuszenie poprawnej promocji, gdy pion wchodzi na ostatni rząd ---
    if ((whitePiece && r2 == 0) || (!whitePiece && r2 == 7)) {
        char p = move.promotion ? std::toupper(move.promotion) : 0;
        if (p != 'Q' && p != 'R' && p != 'B' && p != 'N') {
            std::cout << "DEBUG: Invalid promotion, returning false" << std::endl;
            return false; // musi być podana litera promocji: Q/R/B/N
        }
    }


    // --- SYMULACJA ruchu (uwzględnij usunięcie piona przy EP) ---
    char captured = 0;
    if (isEnPassantCapture) {
        captured = board[epCapRow][epCapCol];
        board[epCapRow][epCapCol] = 0; // tymczasowo zdejmij pion do weryfikacji szacha
    } else {
        captured = board[r2][c2];
    }

    board[r2][c2] = piece;
    board[r1][c1] = 0;

    char kingChar = whitePiece ? 'K' : 'k';
    int kr = 0, kc = 0;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            if (board[r][c] == kingChar) { kr = r; kc = c; }

    bool inCheck = isSquareAttacked(kr, kc, whitePiece ? 'b' : 'w');

    // cofnięcie symulacji
    board[r1][c1] = piece;
    board[r2][c2] = isEnPassantCapture ? 0 : captured;
    if (isEnPassantCapture) {
        board[epCapRow][epCapCol] = captured;
    }

    std::cout << "DEBUG: Pawn move simulation complete, inCheck=" << inCheck << ", returning " << (!inCheck ? "true" : "false") << std::endl;
    return !inCheck;
}

    case 'N':
        std::cout << "DEBUG: Knight move logic" << std::endl;
        if (!((adr == 2 && adc == 1) || (adr == 1 && adc == 2))) {
            std::cout << "DEBUG: Invalid knight move pattern, returning false" << std::endl;
            return false;
        }
        break;
    case 'B':
        std::cout << "DEBUG: Bishop move logic" << std::endl;
        if (adr != adc) {
            std::cout << "DEBUG: Invalid bishop move pattern, returning false" << std::endl;
            return false;
        }
        if (!isPathClear(r1, c1, r2, c2)) {
            std::cout << "DEBUG: Path not clear for bishop, returning false" << std::endl;
            return false;
        }
        break;
    case 'R':
        std::cout << "DEBUG: Rook move logic" << std::endl;
        if (!(adr == 0 || adc == 0)) {
            std::cout << "DEBUG: Invalid rook move pattern, returning false" << std::endl;
            return false;
        }
        if (!isPathClear(r1, c1, r2, c2)) {
            std::cout << "DEBUG: Path not clear for rook, returning false" << std::endl;
            return false;
        }
        break;
    case 'Q':
        std::cout << "DEBUG: Queen move logic" << std::endl;
        if (!(adr == adc || adr == 0 || adc == 0)) {
            std::cout << "DEBUG: Invalid queen move pattern, returning false" << std::endl;
            return false;
        }
        if (!isPathClear(r1, c1, r2, c2)) {
            std::cout << "DEBUG: Path not clear for queen, returning false" << std::endl;
            return false;
        }
        break;
    case 'K':
        {
            std::cout << "DEBUG: King move logic" << std::endl;

            if (adr <= 1 && adc <= 1) {
                std::cout << "DEBUG: Regular king move, OK" << std::endl;
                break;
            }

            if (adr == 0 && adc == 2) {
                std::string reason = checkCastlingReason(move);
                if (reason == "Roszada jest mozliwa.") {
                    std::cout << "DEBUG: Valid castling move, OK" << std::endl;
                    break;
                }
            }
            std::cout << "DEBUG: Invalid king move pattern, returning false" << std::endl;
            return false;
        }

    default:
        std::cout << "DEBUG: Unknown piece type, returning false" << std::endl;
        return false;
    }

    // Sprawdź czy nie bijemy swojego (dla figur innych niż pionek)
    std::cout << "DEBUG: Final check for capturing own piece" << std::endl;
    std::cout << "DEBUG: board[r2][c2] = '" << board[r2][c2] << "'" << std::endl;
    std::cout << "DEBUG: std::isupper(board[r2][c2]) = " << (board[r2][c2] ? std::to_string(std::isupper(board[r2][c2])) : "0") << std::endl;
    std::cout << "DEBUG: whitePiece = " << whitePiece << std::endl;
    if (board[r2][c2] && (std::isupper(board[r2][c2]) == whitePiece)) {
        std::cout << "DEBUG: Trying to capture own piece (final check), returning false" << std::endl;
        return false;
    }

    char captured = board[r2][c2];
    board[r2][c2] = piece;
    board[r1][c1] = 0;

    char kingChar = whitePiece ? 'K' : 'k';
    int kr = 0, kc = 0;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            if (board[r][c] == kingChar)
            {
                kr = r;
                kc = c;
            }

    bool inCheck = isSquareAttacked(kr, kc, whitePiece ? 'b' : 'w');

    board[r1][c1] = piece;
    board[r2][c2] = captured;

    std::cout << "DEBUG: Move simulation complete, inCheck=" << inCheck << ", returning " << (!inCheck ? "true" : "false") << std::endl;
    return !inCheck;
}