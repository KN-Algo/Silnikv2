#include "chess/ai/ZobristHash.h"
#include <random>

uint64_t ZobristHash::pieceKeys[12][8][8];
uint64_t ZobristHash::sideToMoveKey;
uint64_t ZobristHash::castlingKeys[16];
uint64_t ZobristHash::enPassantKeys[8];
bool ZobristHash::initialized = false;

void ZobristHash::initialize() {
    if (initialized) return;
    
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    
    // Inicjalizacja kluczy dla figur
    for (int piece = 0; piece < 12; piece++) {
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                pieceKeys[piece][row][col] = dis(gen);
            }
        }
    }
    
    // Klucz dla strony do ruchu
    sideToMoveKey = dis(gen);
    
    // Klucze dla roszad
    for (int i = 0; i < 16; i++) {
        castlingKeys[i] = dis(gen);
    }
    
    // Klucze dla en passant
    for (int i = 0; i < 8; i++) {
        enPassantKeys[i] = dis(gen);
    }
    
    initialized = true;
}

uint64_t ZobristHash::calculateHash(const char board[8][8], char activeColor, 
                                   const std::string& castling, const std::string& enPassant) {
    if (!initialized) initialize();
    
    uint64_t hash = 0;
    
    // Hash dla figur na planszy
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            char piece = board[row][col];
            if (!piece) continue;
            
            int pieceIndex = 0;
            if (std::isupper(piece)) {
                // Białe figury: 0-5 (P,N,B,R,Q,K)
                switch (piece) {
                    case 'P': pieceIndex = 0; break;
                    case 'N': pieceIndex = 1; break;
                    case 'B': pieceIndex = 2; break;
                    case 'R': pieceIndex = 3; break;
                    case 'Q': pieceIndex = 4; break;
                    case 'K': pieceIndex = 5; break;
                }
            } else {
                // Czarne figury: 6-11 (p,n,b,r,q,k)
                switch (piece) {
                    case 'p': pieceIndex = 6; break;
                    case 'n': pieceIndex = 7; break;
                    case 'b': pieceIndex = 8; break;
                    case 'r': pieceIndex = 9; break;
                    case 'q': pieceIndex = 10; break;
                    case 'k': pieceIndex = 11; break;
                }
            }
            
            hash ^= pieceKeys[pieceIndex][row][col];
        }
    }
    
    // Hash dla strony do ruchu
    if (activeColor == 'b') {
        hash ^= sideToMoveKey;
    }
    
    // Hash dla roszad
    int castlingIndex = 0;
    if (castling.find('K') != std::string::npos) castlingIndex |= 1;
    if (castling.find('Q') != std::string::npos) castlingIndex |= 2;
    if (castling.find('k') != std::string::npos) castlingIndex |= 4;
    if (castling.find('q') != std::string::npos) castlingIndex |= 8;
    hash ^= castlingKeys[castlingIndex];
    
    // Hash dla en passant
    if (enPassant != "-" && enPassant.length() >= 2) {
        int file = enPassant[0] - 'a';
        hash ^= enPassantKeys[file];
    }
    
    return hash;
}

uint64_t ZobristHash::updateHash(uint64_t currentHash, const char board[8][8], 
                                char activeColor, const std::string& castling, 
                                const std::string& enPassant) {
    // Dla uproszczenia, przelicz hash od nowa
    // W rzeczywistej implementacji można by to zoptymalizować
    return calculateHash(board, activeColor, castling, enPassant);
}

uint64_t ZobristHash::getRandomUint64() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    return dis(gen);
}
