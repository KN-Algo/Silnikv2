// Notation.h
#pragma once
#include <string>
#include <cctype>

namespace notation {

    // a1..h8  <-> (row,col)
    inline std::string coordToAlg(int row, int col) {
        char file = 'a' + col;
        char rank = '8' - row;
        return std::string{file} + rank;
    }

    inline bool algToCoord(const std::string& s, int& row, int& col) {
        if (s.size() != 2) return false;
        char f = std::tolower(s[0]), r = s[1];
        if (f < 'a' || f > 'h' || r < '1' || r > '8') return false;
        col = f - 'a';
        row = '8' - r;
        return true;
    }

    // (opcjonalnie, jeśli gdzieś użyjesz)
    inline int fileToCol(char file) { return file - 'a'; }
    inline int rankToRow(char rank) { return '8' - rank; }

} // namespace notation
