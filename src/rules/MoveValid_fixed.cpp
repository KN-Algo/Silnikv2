// MoveValid.cpp
#include "chess/rules/MoveValid.h"
#include "chess/rules/Attack.h"
#include "chess/rules/Castling.h"
#include "chess/utils/Notation.h"
#include <algorithm>
#include <cctype>

using namespace notation;

bool Board::isMoveValid(const Move &move)
{
    int r1 = move.fromRow;
    int c1 = move.fromCol;
    int r2 = move.toRow;
    int c2 = move.toCol;

    char piece = board[r1][c1];
    if (!piece)
        return false;

    bool whitePiece = std::isupper(piece);
    if ((activeColor == 'w' && !whitePiece) || (activeColor == 'b' && whitePiece))
        return false;

    if (board[r2][c2] && ((std::isupper(board[r2][c2]) != 0) == whitePiece))
        return false;

    int dr = r2 - r1;
    int dc = c2 - c1;
    int adr = std::abs(dr), adc = std::abs(dc);

    switch (std::toupper(piece))
    {
    case 'P':
{
    int dir = whitePiece ? -1 : 1; // białe idą w górę (r--), czarne w dół (r++)
    bool isEnPassantCapture = false;
    int epCapRow = -1, epCapCol = -1;

    if (dc == 0) {
        // zwykły ruch do przodu o 1
        if (dr == dir && board[r2][c2] == 0) {
            // OK
        }
        // ruch o 2 z rzędu startowego (6 dla białych, 1 dla czarnych)
        else if (( (whitePiece && r1 == 6) || (!whitePiece && r1 == 1) ) &&
                 dr == 2 * dir &&
                 board[r1 + dir][c1] == 0 && board[r2][c2] == 0)
        {
            // OK (EP target ustawimy w makeMove)
        }
        else {
            return false;
        }
    }
    else if (adr == 1 && dr == dir) {
        // bicie po skosie o 1: zwykłe lub en passant
        if (board[r2][c2] != 0) {
            // zwykłe bicie – sprawdź czy bijemy przeciwnika
            if ((std::isupper(board[r2][c2]) != 0) == whitePiece)
                return false; // nie można bić swojego
        } else {
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
                        isEnPassantCapture = true;
                        epCapRow = capRow; epCapCol = capCol;
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
    }
    else {
        return false;
    }

    // --- wymuszenie poprawnej promocji, gdy pion wchodzi na ostatni rząd ---
    if ((whitePiece && r2 == 0) || (!whitePiece && r2 == 7)) {
        char p = move.promotion ? std::toupper(move.promotion) : 0;
        if (p != 'Q' && p != 'R' && p != 'B' && p != 'N') {
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

    return !inCheck;
}

    case 'N':
        if (!((adr == 2 && adc == 1) || (adr == 1 && adc == 2)))
            return false;
        break;
    case 'B':
        if (adr != adc)
            return false;
        if (!isPathClear(r1, c1, r2, c2))
            return false;
        break;
    case 'R':
        if (!(adr == 0 || adc == 0))
            return false;
        if (!isPathClear(r1, c1, r2, c2))
            return false;
        break;
    case 'Q':
        if (!(adr == adc || adr == 0 || adc == 0))
            return false;
        if (!isPathClear(r1, c1, r2, c2))
            return false;
        break;
    case 'K':
        {

            if (adr <= 1 && adc <= 1) {
                break;
            }

            if (adr == 0 && adc == 2) {
                std::string reason = checkCastlingReason(move);
                if (reason == "Roszada jest mozliwa.") {
                    break;
                }
            }
            return false;
        }

    default:
        return false;
    }

    // Sprawdź czy nie bijemy swojego (dla figur innych niż pionek)
    if (board[r2][c2] && ((std::isupper(board[r2][c2]) != 0) == whitePiece))
        return false;

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

    return !inCheck;
}