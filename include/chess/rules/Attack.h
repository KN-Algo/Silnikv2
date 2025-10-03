// Attack.h
#pragma once
#include "chess/board/Board.h"

namespace Attack {
    bool isPathClear(const char board[8][8], int r1, int c1, int r2, int c2);
    bool isSquareAttacked(const char board[8][8], int row, int col, char byColor);
}


