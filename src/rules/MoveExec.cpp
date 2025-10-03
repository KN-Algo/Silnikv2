// MoveExec.cpp
#include "chess/rules/MoveExec.h"
#include "chess/rules/MoveValid.h"
#include "chess/utils/Notation.h"
#include <cctype>
#include <iostream>

using namespace notation;

void Board::makeMove(const Move &move)
{
    if (!isMoveValid(move))
    {
        std::cout << "Illegal move!\n";
        return;
    }

    char moved = board[move.fromRow][move.fromCol];
    char captured = board[move.toRow][move.toCol];
    bool whiteMoved = std::isupper(moved);

    // Domyślnie brak nowego EP; ustawimy tylko po podwójnym ruchu piona.
    std::string newEnPassant = "-";

    // --- Aktualizacja praw roszady (castling) ---
    auto dropRight = [&](char r) {
        if (castling == "-") return;
        size_t p = castling.find(r);
        if (p != std::string::npos) {
            castling.erase(p, 1);
            if (castling.empty()) castling = "-";
        }
    };

    // 1) Poruszył się KRÓL -> tracimy oba prawa tej strony
    if (std::toupper(moved) == 'K') {
        if (whiteMoved) { dropRight('K'); dropRight('Q'); }
        else            { dropRight('k'); dropRight('q'); }
    }

    // 2) Poruszyła się WIEŻA ze startowego rogu -> tracimy odpowiednie prawo
    if (moved == 'R') {
        if (move.fromRow == 7 && move.fromCol == 7) dropRight('K'); // h1
        if (move.fromRow == 7 && move.fromCol == 0) dropRight('Q'); // a1
    }
    if (moved == 'r') {
        if (move.fromRow == 0 && move.fromCol == 7) dropRight('k'); // h8
        if (move.fromRow == 0 && move.fromCol == 0) dropRight('q'); // a8
    }

    // 3) Zbicie wieży przeciwnika ze startowego rogu -> oni tracą prawo
    if (captured == 'R') {
        if (move.toRow == 7 && move.toCol == 7) dropRight('K'); // białe O-O
        if (move.toRow == 7 && move.toCol == 0) dropRight('Q'); // białe O-O-O
    }
    if (captured == 'r') {
        if (move.toRow == 0 && move.toCol == 7) dropRight('k'); // czarne O-O
        if (move.toRow == 0 && move.toCol == 0) dropRight('q'); // czarne O-O-O
    }

    // --- Przypadki specjalne ---
    bool isCastle = (std::toupper(moved) == 'K' && std::abs(move.toCol - move.fromCol) == 2);
    bool isPawn   = (std::toupper(moved) == 'P');

    // En passant: czy to EP? (cel pusty, skos o 1 i enPassant wskazuje na cel)
    bool isEnPassantCapture = false;
    int dir = whiteMoved ? -1 : 1;
    if (isPawn) {
        int dr = move.toRow - move.fromRow;
        int dc = move.toCol - move.fromCol;
        int adr = std::abs(dr), adc = std::abs(dc);
        if (adc == 1 && dr == dir && captured == 0 && enPassant != "-") {
            int epRow, epCol;
            if (algToCoord(enPassant, epRow, epCol) && epRow == move.toRow && epCol == move.toCol) {
                isEnPassantCapture = true;
            }
        }
    }

    // Jeśli EP – zdejmij piona z „miniętego” pola (za celem)
    if (isEnPassantCapture) {
        int capRow = move.toRow - dir; // pion do zbicia stoi „za” polem celu
        int capCol = move.toCol;
        captured = board[capRow][capCol]; // dla licznika półruchów
        board[capRow][capCol] = 0;
    }

    // --- Wykonanie ruchu figury ---
    board[move.toRow][move.toCol] = moved;
    board[move.fromRow][move.fromCol] = 0;

    // --- Roszada: przesuń wieżę ---
    if (isCastle) {
        int row = move.toRow;
        if (move.toCol == 6) {            // O-O
            board[row][5] = board[row][7];
            board[row][7] = 0;
        } else if (move.toCol == 2) {     // O-O-O
            board[row][3] = board[row][0];
            board[row][0] = 0;
        }
    }

    // --- Promocja piona (domyślnie do hetmana, jeśli brak litery) ---
    if (isPawn && (move.toRow == 0 || move.toRow == 7)) {
        char promo = move.promotion ? move.promotion : 'Q'; // N,B,R,Q
        if (!whiteMoved) promo = std::tolower(promo);
        board[move.toRow][move.toCol] = promo;
    }

    // --- Ustawianie enPassant po podwójnym ruchu piona ---
    if (isPawn) {
        // jeśli to był ruch o 2 pola – EP to pole „minięte”
        if ((whiteMoved && move.fromRow == 6 && move.toRow == 4) ||
            (!whiteMoved && move.fromRow == 1 && move.toRow == 3))
        {
            int passRow = move.fromRow + (whiteMoved ? -1 : 1);
            int passCol = move.fromCol;
            newEnPassant = coordToAlg(passRow, passCol);
        }
    }

    // --- Liczniki & enPassant ---
    // 50‑move rule: reset po ruchu pionem lub biciu (także EP), inaczej +1
    if (isPawn || captured != 0 || isEnPassantCapture) halfmoveClock = 0;
    else halfmoveClock += 1;

    // enPassant: ustaw nowe (po 2‑polowym ruchu piona), w innym wypadku "-"
    enPassant = newEnPassant;

    // fullmoveNumber: po ruchu czarnych zwiększ o 1
    char moverBefore = activeColor;            // kto właśnie się ruszał
    activeColor = (activeColor == 'w') ? 'b' : 'w';
    if (moverBefore == 'b') {
        fullmoveNumber += 1;
    }

    // Zapis pozycji po ruchu (dla 3x powtórzenia)
    gameStateManager.addPosition(board, activeColor, castling, enPassant);

}