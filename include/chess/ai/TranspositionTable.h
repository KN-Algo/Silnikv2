#pragma once
#include <unordered_map>
#include <string>
#include <cstdint>

enum class NodeType {
    EXACT,      // Dokładna wartość
    ALPHA,      // Górna granica (alpha cutoff)
    BETA        // Dolna granica (beta cutoff)
};

struct TranspositionEntry {
    int depth;
    int score;
    NodeType nodeType;
    uint64_t zobristHash;
    
    TranspositionEntry() : depth(0), score(0), nodeType(NodeType::EXACT), zobristHash(0) {}
    TranspositionEntry(int d, int s, NodeType nt, uint64_t zh) 
        : depth(d), score(s), nodeType(nt), zobristHash(zh) {}
};

class TranspositionTable {
private:
    std::unordered_map<uint64_t, TranspositionEntry> table;
    static constexpr size_t MAX_SIZE = 1000000; // Maksymalny rozmiar tablicy
    
public:
    void store(uint64_t hash, int depth, int score, NodeType nodeType);
    bool probe(uint64_t hash, int depth, int& score, NodeType& nodeType);
    void clear();
    size_t size() const { return table.size(); }
};
