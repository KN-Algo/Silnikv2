#include "chess/ai/TranspositionTable.h"
#include <algorithm>

void TranspositionTable::store(uint64_t hash, int depth, int score, NodeType nodeType) {
    // Jeśli tablica jest pełna, usuń najstarsze wpisy
    if (table.size() >= MAX_SIZE) {
        // Usuń wpisy o najmniejszej głębokości
        auto it = std::min_element(table.begin(), table.end(),
            [](const auto& a, const auto& b) {
                return a.second.depth < b.second.depth;
            });
        if (it != table.end()) {
            table.erase(it);
        }
    }
    
    table[hash] = TranspositionEntry(depth, score, nodeType, hash);
}

bool TranspositionTable::probe(uint64_t hash, int depth, int& score, NodeType& nodeType) {
    auto it = table.find(hash);
    if (it == table.end()) {
        return false;
    }
    
    const TranspositionEntry& entry = it->second;
    
    // Zwróć wpis tylko jeśli ma wystarczającą głębokość
    if (entry.depth >= depth) {
        score = entry.score;
        nodeType = entry.nodeType;
        return true;
    }
    
    return false;
}

void TranspositionTable::clear() {
    table.clear();
}
