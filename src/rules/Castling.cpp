// Castling.cpp
#include "chess/rules/Castling.h"
#include "chess/rules/Attack.h" // isSquareAttacked

std::string Board::checkCastlingReason(const Move &move) const // -> Dodane MS
{
    // 0) Czy to w ogóle wygląda jak roszada? (król o 2 w bok, bez zmiany rzędu)
    char piece = board[move.fromRow][move.fromCol];
    if (!piece || std::toupper(piece) != 'K')
        return "To nie jest roszada: z pola startowego nie rusza sie krol.";
    if ((move.toRow - move.fromRow) != 0 || std::abs(move.toCol - move.fromCol) != 2)
        return "To nie jest roszada: krol nie przesuwa sie o 2 pola w bok.";

    bool white = std::isupper(piece);
    int startRow = white ? 7 : 0;
    int startCol = 4;

    if (move.fromRow != startRow || move.fromCol != startCol)
        return "Krol nie stoi na polu startowym (e1/e8).";

    int dc = move.toCol - move.fromCol;
    bool shortSide = (dc == 2); // O-O
    bool longSide = (dc == -2); // O-O-O
    if (!shortSide && !longSide)
        return "To nie jest roszada: zly wektor ruchu.";

    // 1) Prawa do roszady z FEN (KQkq)
    auto hasRight = [&](char r)
    {
        return castling != "-" && castling.find(r) != std::string::npos;
    };
    if (white)
    {
        if (shortSide && !hasRight('K'))
            return "Brak prawa do krotkiej roszady (krol lub wieza ruszali sie wczesniej).";
        if (longSide && !hasRight('Q'))
            return "Brak prawa do dlugiej roszady (krol lub wieza ruszali sie wczesniej).";
    }
    else
    {
        if (shortSide && !hasRight('k'))
            return "Brak prawa do krótkiej roszady (krol lub wieza ruszali sie wczesniej).";
        if (longSide && !hasRight('q'))
            return "Brak prawa do dlugiej roszady (krol lub wieza ruszali sie wczesniej).";
    }

    // 2) Czy odpowiednia wieża stoi na miejscu
    int rookFromCol = shortSide ? 7 : 0;
    char rookChar = white ? 'R' : 'r';
    if (board[startRow][rookFromCol] != rookChar)
        return "Brak wlasciwej wiezy na rogu (a1/h1 lub a8/h8).";

    // 3) Czy między królem a wieżą jest pusto
    int c1 = std::min(startCol, rookFromCol) + 1;
    int c2 = std::max(startCol, rookFromCol) - 1;
    for (int c = c1; c <= c2; ++c)
    {
        if (board[startRow][c] != 0)
            return "Pomiedzy krolem a wieza stoja figury.";
    }

    // 4) Szach na królu teraz?
    char opp = white ? 'b' : 'w';
    if (isSquareAttacked(startRow, startCol, opp))
        return "Krol jest w szachu – roszada zakazana.";

    // 5) Pole pośrednie (f1/d1 lub f8/d8) nie może być atakowane
    int midCol = shortSide ? 5 : 3;
    if (isSquareAttacked(startRow, midCol, opp))
        return "Pole, przez które przechodzi krol, jest atakowane.";

    // 6) Pole docelowe króla (g1/c1 lub g8/c8) nie może być atakowane
    if (isSquareAttacked(startRow, move.toCol, opp))
        return "Pole docelowe krola jest atakowane.";

    return "Roszada jest mozliwa.";
}

