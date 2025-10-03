# Specyfikacja Architektury NNUE - HalfKP

## Przegląd

**Typ Architektury:** HalfKP (Half King-Piece)  
**Platforma docelowa:** Raspberry Pi 4 (8GB RAM)  
**Framework:** PyTorch  
**Silnik inference:** ONNX Runtime (zoptymalizowany pod ARM)

---

## 1. Warstwa wejściowa - Reprezentacja cech

### Definicja przestrzeni cech

#### Cechy HalfKP
Dla każdej pozycji króla (Biały i Czarny osobno):

```
Indeks cechy = pozycja_króla × 640 + typ_figury × 64 + pozycja_figury

Komponenty:
- pozycja_króla: 0-63 (pozycja króla)
- typ_figury: 0-9 (10 typów figur, bez króli)
- pozycja_figury: 0-63 (pozycja figury)
```

#### Kodowanie typów figur

| Indeks | Typ figury        |
|--------|-------------------|
| 0      | Biały pion        |
| 1      | Biały skoczek     |
| 2      | Biały goniec      |
| 3      | Biała wieża       |
| 4      | Biała hetman      |
| 5      | Czarny pion       |
| 6      | Czarny skoczek    |
| 7      | Czarny goniec     |
| 8      | Czarna wieża      |
| 9      | Czarna hetman     |

#### Rozmiar przestrzeni cech

- **Na króla:** 64 × 10 × 64 = 40,960 możliwych cech
- **Całkowita przestrzeń:** 40,960 × 2 = 81,920 cech
- **Aktywne cechy:** ~30-40 jednocześnie (liczba figur na szachownicy)
- **Typ wejścia:** Rzadki (sparse) wektor binarny

---

## 2. Architektura sieci

### Kompletny stos warstw

```
┌─────────────────────────────────────────────────────────────┐
│                    PERSPEKTYWA BIAŁYCH                       │
│  Wejście: Sparse[40960]                                      │
│           Aktywne indeksy: ~30-40                            │
│                          ↓                                   │
│  Feature Transformer: Linear(40960 → 128)                   │
│         Wagi: [40960 × 128] = 5,242,880 param               │
│                          ↓                                   │
│  Akumulator[128]                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    PERSPEKTYWA CZARNYCH                      │
│  Wejście: Sparse[40960]                                      │
│           Aktywne indeksy: ~30-40                            │
│                          ↓                                   │
│  Feature Transformer: Linear(40960 → 128)                   │
│         Wagi: [40960 × 128] = 5,242,880 param               │
│                          ↓                                   │
│  Akumulator[128]                                             │
└─────────────────────────────────────────────────────────────┘

                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      KONKATENACJA                            │
│              [128 + 128] = [256]                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Warstwa 1: Linear(256 → 16)                                 │
│             Wagi: [256 × 16] = 4,096 param                  │
│             Bias: [16]                                       │
│                          ↓                                   │
│  ClippedReLU(x) = max(0, min(1, x))                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Warstwa 2: Linear(16 → 16)                                  │
│             Wagi: [16 × 16] = 256 param                     │
│             Bias: [16]                                       │
│                          ↓                                   │
│  ClippedReLU(x) = max(0, min(1, x))                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Warstwa wyjściowa: Linear(16 → 1)                           │
│                     Wagi: [16 × 1] = 16 param               │
│                     Bias: [1]                                │
│                          ↓                                   │
│  Surowy wynik (centipiony)                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Specyfikacja warstwa po warstwie

### Warstwa 0: Feature Transformer (Perspektywa białych)

| Właściwość | Wartość |
|------------|---------|
| **Kształt wejścia** | [40960] (sparse) |
| **Kształt wyjścia** | [128] |
| **Typ** | Fully Connected (Linear) |
| **Wagi** | 40,960 × 128 = 5,242,880 |
| **Bias** | 128 |
| **Aktywacja** | Brak (akumulator) |
| **Suma parametrów** | 5,243,008 |

**Uwaga:** Ta warstwa jest ewaluowana inkrementalnie przy użyciu aktualizacji akumulatora.

### Warstwa 0': Feature Transformer (Perspektywa czarnych)

| Właściwość | Wartość |
|------------|---------|
| **Kształt wejścia** | [40960] (sparse) |
| **Kształt wyjścia** | [128] |
| **Typ** | Fully Connected (Linear) |
| **Wagi** | 40,960 × 128 = 5,242,880 |
| **Bias** | 128 |
| **Aktywacja** | Brak (akumulator) |
| **Suma parametrów** | 5,243,008 |

**Uwaga:** Identyczna architektura jak dla perspektywy białych, ale trenowana osobno. Cechy są obliczane z punktu widzenia czarnych (szachownica odbita pionowo, kolory odwrócone).

### Warstwa konkatenacji

| Właściwość | Wartość |
|------------|---------|
| **Wejście** | Białe[128] + Czarne[128] |
| **Kształt wyjścia** | [256] |
| **Typ** | Konkatenacja |
| **Parametry** | 0 |

### Warstwa 1: Pierwsza warstwa ukryta

| Właściwość | Wartość |
|------------|---------|
| **Kształt wejścia** | [256] |
| **Kształt wyjścia** | [16] |
| **Typ** | Fully Connected (Linear) |
| **Wagi** | 256 × 16 = 4,096 |
| **Bias** | 16 |
| **Aktywacja** | ClippedReLU |
| **Suma parametrów** | 4,112 |

**ClippedReLU:** `f(x) = max(0, min(1, x))`

### Warstwa 2: Druga warstwa ukryta

| Właściwość | Wartość |
|------------|---------|
| **Kształt wejścia** | [16] |
| **Kształt wyjścia** | [16] |
| **Typ** | Fully Connected (Linear) |
| **Wagi** | 16 × 16 = 256 |
| **Bias** | 16 |
| **Aktywacja** | ClippedReLU |
| **Suma parametrów** | 272 |

**ClippedReLU** 

### Warstwa 3: Warstwa wyjściowa

| Właściwość | Wartość |
|------------|---------|
| **Kształt wejścia** | [16] |
| **Kształt wyjścia** | [1] |
| **Typ** | Fully Connected (Linear) |
| **Wagi** | 16 × 1 = 16 |
| **Bias** | 1 |
| **Aktywacja** | Brak (liniowa) |
| **Suma parametrów** | 17 |

**Interpretacja wyjścia:** Surowy wynik w centipionach (np. 100 = przewaga +1 piona dla białych)

---

## 4. Statystyki modelu

### Liczba parametrów

| Komponent | Parametry |
|-----------|-----------|
| Feature Transformer (Białe) | 5,243,008 |
| Feature Transformer (Czarne) | 5,243,008 |
| Warstwa ukryta 1 | 4,112 |
| Warstwa ukryta 2 | 272 |
| Warstwa wyjściowa | 17 |
| **SUMA** | **10,490,417** (~10.5M parametrów) |

### Zużycie pamięci

| Precyzja | Rozmiar |
|----------|---------|
| Float32 | ~42 MB |
| Float16 | ~21 MB |
| Int8 (Skwantyzowany) | ~10.5 MB |

**Rekomendacja dla RPi:** Kwantyzacja Int8

---

## 5. Funkcje aktywacji

### ClippedReLU

```
f(x) = {
  0    gdy x < 0
  x    gdy 0 ≤ x ≤ 1
  1    gdy x > 1
}
```

**Właściwości:**
- Ogranicza wyjścia do zakresu [0, 1]
- Zapobiega eksplozji gradientu
- Upraszcza kwantyzację do int8
- Używana w Stockfish NNUE


## 6. Obliczenia inkrementalne - Akumulator

### Koncepcja

Zamiast przeliczać cały feature transformer dla każdej pozycji, utrzymujemy **akumulator**, który jest aktualizowany inkrementalnie gdy figury się poruszają.

### Definicja akumulatora

```
akumulator[perspektywa] = Σ (wagi[i] dla i w aktywne_cechy[perspektywa])
```

Gdzie:
- `perspektywa` ∈ {białe, czarne}
- `aktywne_cechy` = indeksy figur na szachownicy względem króla
- `wagi[i]` = 128-wymiarowy wektor z feature transformera

### Reguły aktualizacji

Gdy wykonywany jest ruch:

```
usunięte_cechy = cechy dla figur które się poruszyły/zostały zbite
dodane_cechy = cechy dla figur w nowych pozycjach

akumulator_nowy = akumulator_stary 
                  - Σ wagi[f] dla f w usunięte_cechy
                  + Σ wagi[f] dla f w dodane_cechy
```

**Złożoność:**
- **Pełne obliczenie:** O(30-40) wyszukań cech + sumowanie
- **Aktualizacja inkrementalna:** O(2-4) wyszukań cech + aktualizacje
- **Przyspieszenie:** 10-20x szybciej

### Kiedy odświeżać

Pełne przeliczenie potrzebne gdy:
- Pozycja ustawiana z FEN (nowa pozycja)
- Król się rusza (cechy są względem pozycji króla)
- Przekroczony próg maksymalnej delty (stabilność numeryczna)

---

## 7. Przepływ danych

### Forward Pass (Pełna ewaluacja)

```
1. Wejście: Pozycja (FEN lub bitboardy)
   ↓
2. Ekstrakcja cech:
   - cechy_białych = oblicz_cechy(poz_króla_białych, figury)
   - cechy_czarnych = oblicz_cechy(poz_króla_czarnych, figury)
   ↓
3. Feature Transformer:
   - ak_białe = Σ FT_białe.wagi[cechy_białych]
   - ak_czarne = Σ FT_czarne.wagi[cechy_czarnych]
   ↓
4. Konkatenacja: [ak_białe; ak_czarne] → [256]
   ↓
5. Warstwa ukryta 1:
   - h1 = ClippedReLU(W1 × wejście + b1) → [16]
   ↓
6. Warstwa ukryta 2:
   - h2 = ClippedReLU(W2 × h1 + b2) → [16]
   ↓
7. Warstwa wyjściowa:
   - wynik = W3 × h2 + b3 → [1]
   ↓
8. Zwrot: wynik (w centipionach)
```

### Forward Pass (Inkrementalny)

```
1. Istnieją poprzednie akumulatory:
   - ak_białe_stary
   - ak_czarne_stary
   ↓
2. Oblicz deltę cech z ruchu:
   - usunięte_cechy_białe, dodane_cechy_białe
   - usunięte_cechy_czarne, dodane_cechy_czarne
   ↓
3. Aktualizuj akumulatory:
   - ak_białe = ak_białe_stary - Σ FT[usunięte] + Σ FT[dodane]
   - ak_czarne = ak_czarne_stary - Σ FT[usunięte] + Σ FT[dodane]
   ↓
4. Kontynuuj od kroku 4 w pełnej ewaluacji (konkatenacja)
   ↓
5. Zwrot: wynik
```

---


## 8. Interpretacja wyjścia

### Zakres wyniku

- **Surowe wyjście:** Nieograniczone (typowo -10 do +10 po treningu)
- **Interpretacja:** Centipiony (100 = przewaga 1 piona)

### Przykładowe interpretacje

| Wynik | Znaczenie |
|-------|-----------|
| +300  | Białe mają przewagę ~3 pionów w materiale |
| +100  | Białe mają małą przewagę |
| 0     | Pozycja równa |
| -150  | Czarne mają znaczną przewagę |
| +2000 | Białe wygrywają (2 figury przewagi) |


## 9. Alternatywne architektury

### Większa wersja (jeśli pozwala moc obliczeniowa)

```
[40960 → 256] × 2  (Feature Transformery)
      ↓
   [512]            (Konkatenacja)
      ↓
[512 → 32 → 32 → 1] (Warstwy ukryte)

Suma parametrów: ~21M
Pamięć: ~21MB (int8)
Oczekiwana wydajność: +100-150 ELO vs mniejsza wersja
```

### Mniejsza wersja (ultra lekka)

```
[40960 → 64] × 2   (Feature Transformery)
      ↓
   [128]            (Konkatenacja)
      ↓
[128 → 8 → 1]       (Warstwy ukryte)

Suma parametrów: ~2.6M
Pamięć: ~2.6MB (int8)
Oczekiwana wydajność: -100 ELO vs rekomendowana wersja
```

---

