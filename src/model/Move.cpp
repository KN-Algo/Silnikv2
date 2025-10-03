//
// Created by jakub on 06.08.2025.s
//
#include "chess/model/Move.h"

bool Move::operator==(const Move &other) const
{
    return fromRow == other.fromRow &&
           fromCol == other.fromCol &&
           toRow == other.toRow &&
           toCol == other.toCol &&
           movedPiece == other.movedPiece &&
           capturedPiece == other.capturedPiece &&
           promotion == other.promotion;
}

// operator to compare two "different" moves
