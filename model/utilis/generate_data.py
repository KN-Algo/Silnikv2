"""
Kompletny skrypt do przetwarzania PGN z Lichess do datasetu NNUE

Konfiguracja w sekcji __main__ na dole pliku
"""

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm
import os


def fen_to_features(fen):
    """
    Konwertuje FEN do features dla NNUE HalfKP
    
    Returns:
        (white_features, black_features) - listy indeksów aktywnych cech
        lub (None, None) jeśli błąd
    """
    try:
        board = chess.Board(fen)
    except:
        return None, None
    
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    
    if white_king_sq is None or black_king_sq is None:
        return None, None
    
    white_features = []
    black_features = []
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.piece_type == chess.KING:
            continue
        
        # Indeks typu figury (0-9)
        # 0-4: białe (pawn, knight, bishop, rook, queen)
        # 5-9: czarne (pawn, knight, bishop, rook, queen)
        piece_idx = (piece.piece_type - 1)
        if piece.color == chess.BLACK:
            piece_idx += 5
        
        # Feature dla białych (z perspektywy białego króla)
        white_feature = white_king_sq * 640 + piece_idx * 64 + square
        white_features.append(white_feature)
        
        # Feature dla czarnych (odbicie + zamiana kolorów)
        flipped_square = square ^ 56  # Odbij pionowo (XOR z 56)
        flipped_king = black_king_sq ^ 56
        
        # Zamień kolor figury
        if piece.color == chess.WHITE:
            flipped_piece_idx = piece_idx + 5
        else:
            flipped_piece_idx = piece_idx - 5
        
        black_feature = flipped_king * 640 + flipped_piece_idx * 64 + flipped_square
        black_features.append(black_feature)
    
    return white_features, black_features


def should_skip_game(game, min_elo=2000):
    """
    Czy pominąć tę partię? (filtrowanie)
    
    Args:
        game: chess.pgn.Game object
        min_elo: minimalny rating (default: 2000)
    
    Returns:
        True jeśli partię należy pominąć
    """
    headers = game.headers
    
    # 1. Sprawdź wynik
    if headers["Result"] not in ["1-0", "0-1", "1/2-1/2"]:
        return True
    
    # 2. Sprawdź rating - minimum dla OBU graczy
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
        
        if white_elo < min_elo or black_elo < min_elo:
            return True
            
    except (ValueError, TypeError):
        # Brak ELO lub błąd parsowania - odrzuć
        return True
    
    # 3. Sprawdź termination - tylko normalne zakończenie
    termination = headers.get("Termination", "").lower()
    if any(bad in termination for bad in ["time forfeit", "abandoned", "rules infraction"]):
        return True
    
    return False


def process_game(game, positions_per_game=30, skip_opening_moves=10):
    """
    Przetwarza jedną partię i zwraca pozycje treningowe
    
    Args:
        game: chess.pgn.Game object
        positions_per_game: ile pozycji wyciągnąć z jednej partii
        skip_opening_moves: ile pierwszych ruchów pominąć (teoria debiutowa)
    
    Returns:
        Lista słowników z pozycjami
    """
    positions = []
    
    # Pobierz wynik
    result = game.headers["Result"]
    if result == "1-0":
        label = 1.0
    elif result == "0-1":
        label = 0.0
    else:  # 1/2-1/2
        label = 0.5
    
    # Rozegraj partię
    board = game.board()
    moves = list(game.mainline_moves())
    total_moves = len(moves)
    
    # Pomiń za krótkie i za długie partie
    if total_moves < 30 or total_moves > 120:
        return []
    
    # Wybierz które ruchy zapisać (równomiernie rozłożone, pomijając opening)
    moves_to_save = set()
    effective_moves = total_moves - skip_opening_moves
    
    if effective_moves <= positions_per_game:
        moves_to_save = set(range(skip_opening_moves, total_moves))
    else:
        step = effective_moves / positions_per_game
        moves_to_save = {int(skip_opening_moves + i * step) for i in range(positions_per_game)}
    
    # Zbierz pozycje
    for move_idx, move in enumerate(moves):
        board.push(move)
        
        if move_idx in moves_to_save:
            # Pomiń pozycje w szachu (dla uproszczenia)
            if board.is_check():
                continue
            
            # Pomiń pozycje z bardzo małą ilością materiału (tablice końcówkowe)
            piece_count = len(board.piece_map())
            if piece_count < 6:
                continue
            
            fen = board.fen()
            white_feat, black_feat = fen_to_features(fen)
            
            if white_feat is None:
                continue
            
            positions.append({
                'white_features': white_feat,
                'black_features': black_feat,
                'label': label,
                'fen': fen
            })
    
    return positions


def process_pgn_file(pgn_path, output_path, max_games=None, 
                     positions_per_game=30, min_elo=2000,
                     checkpoint_every=10000):
    """
    Przetwarza cały plik PGN
    
    Args:
        pgn_path: ścieżka do pliku PGN
        output_path: gdzie zapisać wynikowy plik .npz
        max_games: maksymalna liczba partii do przetworzenia (None = wszystkie)
        positions_per_game: ile pozycji wyciągnąć z każdej partii
        min_elo: minimalny rating graczy
        checkpoint_every: co ile partii zapisywać checkpoint
    
    Returns:
        Lista wszystkich pozycji
    """
    print(f"Przetwarzanie PGN")
    print(f"   Plik wejściowy: {pgn_path}")
    print(f"   Plik wyjściowy: {output_path}")
    print(f"   Min ELO: {min_elo}")
    print(f"   Max partii: {max_games if max_games else 'wszystkie'}")
    print(f"   Pozycji na partię: {positions_per_game}")
    print()
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(pgn_path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {pgn_path}")
    
    # Utwórz folder wyjściowy jeśli nie istnieje
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_positions = []
    games_processed = 0
    games_skipped = 0
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
        pbar = tqdm(desc="Partie", unit="game")
        
        while True:
            # Sprawdź limit
            if max_games and games_processed >= max_games:
                break
            
            # Wczytaj następną partię
            try:
                game = chess.pgn.read_game(pgn_file)
            except Exception as e:
                print(f"\nBłąd czytania partii: {e}")
                break
            
            if game is None:
                break
            
            # Filtruj
            if should_skip_game(game, min_elo=min_elo):
                games_skipped += 1
                pbar.update(1)
                continue
            
            # Przetwórz
            try:
                positions = process_game(game, positions_per_game)
                all_positions.extend(positions)
                games_processed += 1
            except Exception as e:
                print(f"\nBłąd w partii {games_processed + games_skipped}: {e}")
                games_skipped += 1
            
            # Checkpoint
            if checkpoint_every and games_processed % checkpoint_every == 0 and games_processed > 0:
                checkpoint_path = output_path.replace('.npz', f'_checkpoint_{games_processed}.npz')
                print(f"\n💾 Checkpoint: {checkpoint_path}")
                np.savez_compressed(checkpoint_path, 
                                  positions=np.array(all_positions, dtype=object))
            
            # Aktualizuj progress bar
            pbar.update(1)
            pbar.set_postfix({
                'przetworzono': games_processed,
                'pozycje': len(all_positions),
                'pominięto': games_skipped
            })
        
        pbar.close()
    
    print(f"\nPrzetworzono {games_processed} partii")
    print(f"Pominięto {games_skipped} partii")
    print(f"Wygenerowano {len(all_positions)} pozycji")
    
    # Zapisz finalny plik
    print(f"\nZapisywanie do {output_path}...")
    np.savez_compressed(output_path, positions=np.array(all_positions, dtype=object))
    
    # Sprawdź rozmiar pliku
    file_size = os.path.getsize(output_path) / (1024**3)  # GB
    print(f"Zapisano! Rozmiar: {file_size:.2f} GB")
    
    return all_positions


def get_stats(positions):
    """
    Wyświetla statystyki datasetu
    """
    if len(positions) == 0:
        print("Brak pozycji do analizy!")
        return
    
    labels = [p['label'] for p in positions]
    
    white_wins = sum(1 for l in labels if l == 1.0)
    black_wins = sum(1 for l in labels if l == 0.0)
    draws = sum(1 for l in labels if l == 0.5)
    
    print("\n" + "="*50)
    print("STATYSTYKI DATASETU")
    print("="*50)
    print(f"Wszystkie pozycje: {len(positions):,}")
    print(f"\nRozkład wyników:")
    print(f"  Wygrane białych: {white_wins:,} ({white_wins/len(positions)*100:.1f}%)")
    print(f"  Wygrane czarnych: {black_wins:,} ({black_wins/len(positions)*100:.1f}%)")
    print(f"  Remisy:          {draws:,} ({draws/len(positions)*100:.1f}%)")
    
    # Średnia liczba figur
    avg_white_features = np.mean([len(p['white_features']) for p in positions])
    avg_black_features = np.mean([len(p['black_features']) for p in positions])
    print(f"\nŚrednia liczba figur na szachownicy:")
    print(f"  White features: {avg_white_features:.1f}")
    print(f"  Black features: {avg_black_features:.1f}")
    
    # Przykładowa pozycja
    print(f"\nPrzykładowa pozycja:")
    sample = positions[len(positions)//2]  # Środkowa pozycja
    print(f"  FEN: {sample['fen']}")
    print(f"  Label: {sample['label']} ", end="")
    if sample['label'] == 1.0:
        print("(wygrana białych)")
    elif sample['label'] == 0.0:
        print("(wygrana czarnych)")
    else:
        print("(remis)")
    
    print("="*50)




if __name__ == "__main__":
    
    PGN_PATH = "model/data/raw/lichess_db_standard_rated_2024-10.pgn"
    
    OUTPUT_PATH = "model/data/processed/positions_elo2000.npz"
    
    # Minimalny ELO graczy
    MIN_ELO = 2000
    
    # Maksymalna liczba partii do przetworzenia
    # None = wszystkie partie z pliku
    # 50000 = tylko 50k partii (dla szybkiego testu)
    MAX_GAMES = None 
    
    # Ile pozycji wyciągnąć z każdej partii
    POSITIONS_PER_GAME = 30
    
    # checkpoint (na wypadek crash)
    CHECKPOINT_EVERY = 10000
    
    print("="*60)
    print(" "*15 + "NNUE DATASET GENERATOR")
    print("="*60)
    print()
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(PGN_PATH):
        print(f"   BŁĄD: Nie znaleziono pliku PGN!")
        print(f"   Sprawdź ścieżkę: {PGN_PATH}")
        print()
        print("   Upewnij się że:")
        print("   1. Pobrałeś plik z Lichess")
        print("   2. Rozpakowałeś .zst (zstd -d plik.pgn.zst)")
        print("   3. Ścieżka w PGN_PATH jest poprawna")
        exit(1)
    
    # Wyświetl rozmiar pliku
    file_size_gb = os.path.getsize(PGN_PATH) / (1024**3)
    print(f"lik PGN: {file_size_gb:.1f} GB")
    print()
    
    # Przetwórz
    try:
        positions = process_pgn_file(
            pgn_path=PGN_PATH,
            output_path=OUTPUT_PATH,
            max_games=MAX_GAMES,
            positions_per_game=POSITIONS_PER_GAME,
            min_elo=MIN_ELO,
            checkpoint_every=CHECKPOINT_EVERY
        )
        
        # Statystyki
        get_stats(positions)
        
        print("\nGOTOWE!")
        print(f"   Dataset zapisany w: {OUTPUT_PATH}")
        print()
        print("Następne kroki:")
        print("  1. Zweryfikuj dataset: python verify_dataset.py")
        print("  2. Trenuj model: python train.py")
        
    except KeyboardInterrupt:
        print("\n\nPrzerwano przez użytkownika (Ctrl+C)")
        print("   Checkpointy zostały zapisane i możesz kontynuować później")
        
    except Exception as e:
        print(f"\n\nBŁĄD: {e}")
        import traceback
        traceback.print_exc()