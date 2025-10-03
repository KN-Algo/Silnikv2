#pragma once
#include <vector>
#include <chrono>
#include "chess/model/Move.h"
#include "chess/ai/TranspositionTable.h"
#include "chess/ai/ZobristHash.h"

struct SearchResult {
    Move bestMove;
    int score;
    int depth;
    uint64_t nodesVisited;
    std::chrono::milliseconds timeSpent;
    
    SearchResult() : score(0), depth(0), nodesVisited(0), timeSpent(0) {}
};

class ChessAI {
private:
    TranspositionTable transpositionTable;
    uint64_t nodesVisited;
    std::chrono::steady_clock::time_point searchStartTime;
    
    // Parametry wyszukiwania
    static constexpr int MAX_DEPTH = 50;
    static constexpr int MAX_TIME_MS = 30000; // 30 sekund
    
    // NegaMax z Alfa-Beta Pruning
    int negamax(const char board[8][8], char activeColor, const std::string& castling, 
                const std::string& enPassant, int depth, int alpha, int beta, 
                uint64_t zobristHash);
    
    // Iterative Deepening
    SearchResult iterativeDeepening(const char board[8][8], char activeColor, 
                                   const std::string& castling, const std::string& enPassant,
                                   int maxDepth, int maxTimeMs);
    
    // Sprawdzenie czy czas się skończył
    bool isTimeUp() const;
    
    // Sortowanie ruchów dla lepszego Alfa-Beta Pruning
    void sortMoves(std::vector<Move>& moves, const char board[8][8], 
                   char activeColor, const std::string& castling, 
                   const std::string& enPassant);
    
public:
    ChessAI();
    
    // Główna funkcja AI - zwraca najlepszy ruch
    SearchResult findBestMove(const char board[8][8], char activeColor, 
                             const std::string& castling, const std::string& enPassant,
                             int maxDepth = 20, int maxTimeMs = 5000);
    
    // Reset licznika węzłów
    void resetNodesCount() { nodesVisited = 0; }
    
    // Pobierz liczbę odwiedzonych węzłów
    uint64_t getNodesVisited() const { return nodesVisited; }
    
    // Wyczyść tablicę transpozycji
    void clearTranspositionTable() { transpositionTable.clear(); }
};
