#include "chess/ai/ChessAI.h"
#include "chess/rules/MoveGenerator.h"
#include "chess/ai/Evaluator.h"
#include <algorithm>
#include <iostream>
#include <limits>

ChessAI::ChessAI() : nodesVisited(0) {
    ZobristHash::initialize();
}

SearchResult ChessAI::findBestMove(const char board[8][8], char activeColor, 
                                  const std::string& castling, const std::string& enPassant,
                                  int maxDepth, int maxTimeMs) {
    resetNodesCount();
    searchStartTime = std::chrono::steady_clock::now();
    
    return iterativeDeepening(board, activeColor, castling, enPassant, maxDepth, maxTimeMs);
}

SearchResult ChessAI::iterativeDeepening(const char board[8][8], char activeColor, 
                                        const std::string& castling, const std::string& enPassant,
                                        int maxDepth, int maxTimeMs) {
    SearchResult result;
    result.bestMove = {0, 0, 0, 0, '?', 0}; // Domyślny ruch
    
    // Generuj wszystkie legalne ruchy
    std::vector<Move> moves = MoveGenerator::generateLegalMoves(board, activeColor, castling, enPassant);
  
    // Sprawdź liczbę ruchów
    if (moves.empty()) {
        std::cout << "UWAGA: AI nie znalazł żadnych legalnych ruchów!" << std::endl;
        return result;
    }
    
    // Sortuj ruchy dla lepszego Alfa-Beta Pruning
    sortMoves(moves, board, activeColor, castling, enPassant);
    
    // Iterative Deepening - zaczynamy od głębokości 1
    for (int depth = 1; depth <= maxDepth; depth++) {
        if (isTimeUp()) break;
        
        SearchResult currentResult;
        currentResult.depth = depth;
        currentResult.bestMove = moves[0]; // Domyślnie pierwszy ruch
        
        int bestScore = std::numeric_limits<int>::min();
        int alpha = std::numeric_limits<int>::min();
        int beta = std::numeric_limits<int>::max();
        
        // Wyszukaj najlepszy ruch dla aktualnej głębokości
        for (const Move& move : moves) {
            // Symuluj ruch
            char tempBoard[8][8];
            std::string tempCastling = castling;
            std::string tempEnPassant = enPassant;
            char tempActiveColor = activeColor;
            
            // Skopiuj planszę
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    tempBoard[i][j] = board[i][j];
                }
            }
            
            // Wykonaj ruch (uproszczona wersja)
            tempBoard[move.toRow][move.toCol] = tempBoard[move.fromRow][move.fromCol];
            tempBoard[move.fromRow][move.fromCol] = 0;
            
            // Obsługa promocji piona
            if (move.promotion) {
                tempBoard[move.toRow][move.toCol] = move.promotion;
            }

            // Zmień stronę do ruchu
            tempActiveColor = (tempActiveColor == 'w') ? 'b' : 'w';
            
            // Oblicz nowy hash
            uint64_t newHash = ZobristHash::calculateHash(tempBoard, tempActiveColor, tempCastling, tempEnPassant);
            
            // Wykonaj wyszukiwanie NegaMax
            int score = -negamax(tempBoard, tempActiveColor, tempCastling, tempEnPassant, 
                                depth - 1, -beta, -alpha, newHash);
            
            if (score > bestScore) {
                bestScore = score;
                currentResult.bestMove = move;
            }
            
            alpha = std::max(alpha, score);
            if (alpha >= beta) break; // Beta cutoff
        }
        
        currentResult.score = bestScore;
        currentResult.nodesVisited = nodesVisited;
        currentResult.timeSpent = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - searchStartTime);
        
        // Jeśli czas się skończył, przerwij
        if (isTimeUp()) break;
        
        // Zaktualizuj wynik
        result = currentResult;
        
        // Jeśli znaleźliśmy mata, nie ma potrzeby szukać głębiej
        if (std::abs(result.score) > 10000) break;
        
        // Sprawdź czy AI znalazło ruch
        if (result.bestMove.fromRow == 0 && result.bestMove.fromCol == 0 && 
            result.bestMove.toRow == 0 && result.bestMove.toCol == 0) {
            std::cout << "UWAGA: AI nie znalazło żadnego ruchu!" << std::endl;
            break;
        }

        std::cout << "Głębokość " << depth << ": " << result.score
                  << " (węzły: " << result.nodesVisited << ")" << std::endl;
    }
    
    return result;
}

int ChessAI::negamax(const char board[8][8], char activeColor, const std::string& castling, 
                     const std::string& enPassant, int depth, int alpha, int beta, 
                     uint64_t zobristHash) {
    nodesVisited++;
    
    // Sprawdź tablicę transpozycji
    int ttScore;
    NodeType ttNodeType;
    if (transpositionTable.probe(zobristHash, depth, ttScore, ttNodeType)) {
        if (ttNodeType == NodeType::EXACT) {
            return ttScore;
        } else if (ttNodeType == NodeType::ALPHA && ttScore <= alpha) {
            return alpha;
        } else if (ttNodeType == NodeType::BETA && ttScore >= beta) {
            return beta;
        }
    }
    
    // Sprawdź czy czas się skończył
    if (isTimeUp()) {
        return 0;
    }
    
    // Sprawdź czy osiągnęliśmy maksymalną głębokość lub koniec gry
    if (depth == 0) {
        int score = Evaluator::evaluatePosition(board, activeColor);
        transpositionTable.store(zobristHash, depth, score, NodeType::EXACT);
        return score;
    }
    
    // Sprawdź stan gry
    bool hasLegalMoves = MoveGenerator::hasLegalMoves(board, activeColor, castling, enPassant);
    bool isInCheck = MoveGenerator::isInCheck(board, activeColor);
    
    if (!hasLegalMoves) {
        if (isInCheck) {
            // Mat - bardzo duża wartość
            int score = (activeColor == 'w') ? -10000 : 10000;
            transpositionTable.store(zobristHash, depth, score, NodeType::EXACT);
            return score;
        } else {
            // Pat - remis
            transpositionTable.store(zobristHash, depth, 0, NodeType::EXACT);
            return 0;
        }
    }
    
    // Generuj wszystkie legalne ruchy
    std::vector<Move> moves = MoveGenerator::generateLegalMoves(board, activeColor, castling, enPassant);
    
    // Sortuj ruchy dla lepszego Alfa-Beta Pruning
    sortMoves(moves, board, activeColor, castling, enPassant);
    
    int bestScore = std::numeric_limits<int>::min();
    NodeType bestNodeType = NodeType::ALPHA;
    int originalAlpha = alpha;
    
    for (const Move& move : moves) {
        // Symuluj ruch
        char tempBoard[8][8];
        std::string tempCastling = castling;
        std::string tempEnPassant = enPassant;
        char tempActiveColor = activeColor;
        
        // Skopiuj planszę
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                tempBoard[i][j] = board[i][j];
            }
        }
        
        // Wykonaj ruch (uproszczona wersja)
        tempBoard[move.toRow][move.toCol] = tempBoard[move.fromRow][move.fromCol];
        tempBoard[move.fromRow][move.fromCol] = 0;
        
        // Obsługa promocji piona
        if (move.promotion) {
            tempBoard[move.toRow][move.toCol] = move.promotion;
        }
      
        // Zmień stronę do ruchu
        tempActiveColor = (tempActiveColor == 'w') ? 'b' : 'w';
        
        // Oblicz nowy hash
        uint64_t newHash = ZobristHash::calculateHash(tempBoard, tempActiveColor, tempCastling, tempEnPassant);
        
        // Rekurencyjne wywołanie NegaMax
        int score = -negamax(tempBoard, tempActiveColor, tempCastling, tempEnPassant, 
                            depth - 1, -beta, -alpha, newHash);
        
        if (score > bestScore) {
            bestScore = score;
        }
        
        // Alfa-Beta Pruning
        if (score >= beta) {
            transpositionTable.store(zobristHash, depth, score, NodeType::BETA);
            return score;
        }
        
        if (score > alpha) {
            alpha = score;
            bestNodeType = NodeType::EXACT;
        }
    }
    
    // Określ typ węzła dla tablicy transpozycji
    if (bestScore <= originalAlpha) {
        bestNodeType = NodeType::ALPHA;
    } else if (bestScore >= beta) {
        bestNodeType = NodeType::BETA;
    } else {
        bestNodeType = NodeType::EXACT;
    }
    
    // Zapisz wynik w tablicy transpozycji
    transpositionTable.store(zobristHash, depth, bestScore, bestNodeType);
    
    return bestScore;
}

bool ChessAI::isTimeUp() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - searchStartTime);
    return elapsed.count() >= MAX_TIME_MS;
}

void ChessAI::sortMoves(std::vector<Move>& moves, const char board[8][8], 
                        char activeColor, const std::string& castling, 
                        const std::string& enPassant) {
    // Sortuj ruchy według priorytetu:
    // 1. Bicia (z wartością bicia)
    // 2. Ruchy promocji
    // 3. Ruchy do centrum
    // 4. Pozostałe ruchy
    
    std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        int scoreA = 0, scoreB = 0;
        
        // Bicia
        if (a.capturedPiece) {
            scoreA += 1000 + Evaluator::PAWN_VALUE; // Podstawowy bonus za bicie
            switch (std::toupper(a.capturedPiece)) {
                case 'P': scoreA += Evaluator::PAWN_VALUE; break;
                case 'N': scoreA += Evaluator::KNIGHT_VALUE; break;
                case 'B': scoreA += Evaluator::BISHOP_VALUE; break;
                case 'R': scoreA += Evaluator::ROOK_VALUE; break;
                case 'Q': scoreA += Evaluator::QUEEN_VALUE; break;
            }
        }
        if (b.capturedPiece) {
            scoreB += 1000 + Evaluator::PAWN_VALUE;
            switch (std::toupper(b.capturedPiece)) {
                case 'P': scoreB += Evaluator::PAWN_VALUE; break;
                case 'N': scoreB += Evaluator::KNIGHT_VALUE; break;
                case 'B': scoreB += Evaluator::BISHOP_VALUE; break;
                case 'R': scoreB += Evaluator::ROOK_VALUE; break;
                case 'Q': scoreB += Evaluator::QUEEN_VALUE; break;
            }
        }
        
        // Promocje
        if (a.promotion) scoreA += 500;
        if (b.promotion) scoreB += 500;
        
        // Ruchy do centrum
        int centerDistanceA = std::abs(a.toRow - 3.5) + std::abs(a.toCol - 3.5);
        int centerDistanceB = std::abs(b.toRow - 3.5) + std::abs(b.toCol - 3.5);
        scoreA += (7 - centerDistanceA) * 10;
        scoreB += (7 - centerDistanceB) * 10;
        
        return scoreA > scoreB;
    });
}
