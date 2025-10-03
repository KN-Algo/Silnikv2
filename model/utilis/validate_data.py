"""
Weryfikuje wygenerowany dataset
"""

import numpy as np
import chess
import sys

def verify_dataset(npz_path):
    print(f"Weryfikacja datasetu: {npz_path}\n")
    
    # Załaduj
    try:
        data = np.load(npz_path, allow_pickle=True)
        positions = data['positions']
    except Exception as e:
        print(f"Błąd ładowania: {e}")
        return False
    
    print(f"Załadowano {len(positions):,} pozycji\n")
    
    if len(positions) == 0:
        print("Dataset pusty!")
        return False
    
    # Sprawdź strukturę
    print("Sprawdzanie struktury...")
    sample = positions[0]
    
    required_keys = ['white_features', 'black_features', 'label', 'fen']
    for key in required_keys:
        if key not in sample:
            print(f"Brak klucza: {key}")
            return False
    print("Struktura OK\n")
    
    # Sprawdź features
    print("🔍 Sprawdzanie features...")
    errors = 0
    for i in range(min(1000, len(positions))):
        pos = positions[i]
        
        white_feat = pos['white_features']
        black_feat = pos['black_features']
        
        # Sprawdź zakres
        if any(f < 0 or f >= 40960 for f in white_feat):
            print(f"Pozycja {i}: white features poza zakresem [0, 40960)")
            errors += 1
        if any(f < 0 or f >= 40960 for f in black_feat):
            print(f"Pozycja {i}: black features poza zakresem [0, 40960)")
            errors += 1
        
        # Sprawdź label
        if pos['label'] not in [0.0, 0.5, 1.0]:
            print(f"Pozycja {i}: nieprawidłowy label {pos['label']}")
            errors += 1
    
    if errors == 0:
        print("Features OK\n")
    else:
        print(f"Znaleziono {errors} błędów\n")
        return False
    
    # Przykładowe pozycje
    print("Przykładowe pozycje:\n")
    for i in [0, len(positions)//2, -1]:
        pos = positions[i]
        board = chess.Board(pos['fen'])
        label_str = {1.0: "białe", 0.0: "czarne", 0.5: "remis"}[pos['label']]
        print(f"Pozycja {i}:")
        print(board)
        print(f"Wynik: {label_str}\n")
    
    print("Weryfikacja zakończona pomyślnie!")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        npz_path = "data/processed/positions_elo2000_checkpoint_180000.npz"
    
    verify_dataset(npz_path)