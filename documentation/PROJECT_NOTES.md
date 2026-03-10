# ClickBait Detection — Project Notes

## Struktura projekta

```
ClickBait_Detection/
  data/
    train.csv          - 28,800 naslova za trening (90%)
    val.csv            -  2,560 naslova za validaciju (8%)
    test.csv           -    640 naslova za testiranje (2%)
  weights/             - sacuvani weights modela nakon treninga
  runs/                - tensorboard logovi
  config.py            - hiperparametri i putanje
  model.py             - arhitektura modela (Encoder-only Transformer)
  dataset.py           - ucitavanje i obrada podataka
  train.py             - trening loop
  test.py              - validacija i testiranje (funkcije)
  run_test.py          - pokretanje testa bez treninga
  predict.py           - klasifikacija sopstvenih naslova
```

---

## Pokretanje treninga

```bash
cd "D:/Fakultet Master/SIPB/ClickBait_Detection"
python train.py
```

Weights se automatski cuvaju u `weights/` folderu:
- nakon epohe 0
- svakih 10 epoha (9, 19, 29...)
- na kraju treninga (epoha 99)

---

## Pracenje treninga uzivo (TensorBoard)

Dok trening radi, otvori novi terminal i pokreni:

```bash
python -m tensorboard.main --logdir "D:/Fakultet Master/SIPB/ClickBait_Detection/runs"
```

Pa otvori u browseru: `http://localhost:6006`

---

## Sta gledati u TensorBoard-u

| Graf | Sta ocekivati | Problem |
|------|---------------|---------|
| `train_loss` | Konstantno pada | Ako stagnira od pocetka — model ne uci |
| `val_loss` | Pada zajedno sa train_loss | Ako raste dok train_loss pada — overfitting |
| `val_accuracy` | Raste, cilj >85% | Ako zaglavi na ~50% — model ne uci nista |
| `val_f1` | Raste zajedno sa accuracy | Ako mnogo zaostaje za accuracy — model pristrasан ka jednoj klasi |

### Dobar trening izgleda ovako:
```
Epoha  0: loss ~0.60, Accuracy ~70%, F1 ~70%
Epoha 10: loss ~0.20, Accuracy ~90%, F1 ~90%
Epoha 50: loss ~0.05, Accuracy ~98%, F1 ~98%
```

---

## Pokretanje testa nakon treninga

```bash
python run_test.py
```

Ocekivani ispis:
```
Using device: cuda
Loading model: weights\clickbait_detector_99.pt
Model from epoch 99 loaded successfully.

Test set size: 640 headlines

===== TEST RESULTS =====
Accuracy:  0.9850
Precision: 0.9820
Recall:    0.9880
F1 Score:  0.9855
========================
```

### Sta znace metrike:

- **Accuracy** — koliko % naslova je ispravno klasifikovano
- **Precision** — od svih koje je model oznacio kao clickbait, koliko je stvarno clickbait
- **Recall** — od svih stvarnih clickbait naslova, koliko je model pronasao
- **F1** — kombinacija precision i recall, glavna metrika (dataset je balansiran)

### Ocena rezultata:

| F1 Score | Ocena |
|----------|-------|
| ispod 75% | Los |
| 75% - 85% | Prosecno |
| 85% - 92% | Dobro |
| iznad 92% | Odlicno |

---

## Nastavak treninga od mesta gde je stao

Ako prekines trening i hoces da nastavis, u `config.py` promeni:

```python
"preload": "latest",
```

Pa pokreni `python train.py` ponovo.

Nakon sto trening zavrsis, vrati na:

```python
"preload": None,
```

---

## Klasifikacija sopstvenih naslova

Napravi fajl `predict.py`:

```python
from test import classify_headline

headlines = [
    "You Won't Believe What This Dog Did Next",
    "Federal Reserve raises interest rates by 0.25%",
    "15 Reasons Why You're Still Single",
    "NATO summit concludes with new defense agreement",
]

for h in headlines:
    print(classify_headline(h))
    print(f"  -> {h}\n")
```

Pokreni:
```bash
python predict.py
```

---

## Znaci da nesto nije u redu

| Simptom | Uzrok | Resenje |
|---------|-------|---------|
| `loss: nan` od pocetka | learning rate previsok | U config.py smanji na `1e-4` |
| Accuracy zaglavljeno na 50% | model ne uci | Proveri dataset, restart trening |
| `No saved model found` u run_test.py | trening nije zavrsio | Pokreni train.py do kraja |
| `ModuleNotFoundError` | nedostaje paket | `pip install -r requirements.txt` |
| Trening spor (~1.2s/it) | koristi CPU | Instaliraj PyTorch sa CUDA podrskom |
