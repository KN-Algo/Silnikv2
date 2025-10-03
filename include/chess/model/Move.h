//
// Created by jakub on 06.08.2025.s
//

#pragma once

struct Move
{
    int fromRow, fromCol;
    int toRow, toCol;
    char movedPiece;
    char capturedPiece;
    char promotion = 0;

    bool operator==(const Move &other) const;
};