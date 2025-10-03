#pragma once
#include <cstdint>
#include <random>

class ZobristHash {
private:
    static uint64_t pieceKeys[12][8][8];  // 12 typów figur (6 białe + 6 czarne) × 8×8 pól
    static uint64_t sideToMoveKey;        // Klucz dla strony do ruchu
    static uint64_t castlingKeys[16];     // Klucze dla wszystkich kombinacji roszad
    static uint64_t enPassantKeys[8];    // Klucze dla en passant
    
    static bool initialized;
    
public:
    static void initialize();
    static uint64_t calculateHash(const char board[8][8], char activeColor, 
                                 const std::string& castling, const std::string& enPassant);
    static uint64_t updateHash(uint64_t currentHash, const char board[8][8], 
                              char activeColor, const std::string& castling, 
                              const std::string& enPassant);
    
private:
    static uint64_t getRandomUint64();
};
