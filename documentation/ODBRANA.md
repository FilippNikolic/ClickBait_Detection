# Priprema za odbranu — Detekcija Clickbait Naslova

---

## 1. Sta je problem koji resavas?

**Clickbait** je tekstualni sadrzaj ciji je jedini cilj da privuce klik — koristi senzacionalizam,
namerno izostavlja kljucne informacije ili direktno se obraca citaocu. Primeri:
- "You Won't Believe What Happened Next"
- "15 Things Only 90s Kids Remember"

**Problem:** Ovakav sadrzaj narusava korisnicko iskustvo i doprinosi sirenju dezinformacija.

**Resenje:** NLP model koji automatski klasifikuje naslov kao clickbait (1) ili nije clickbait (0).

---

## 2. Koji dataset koristis?

- **Izvor:** Kaggle — Clickbait Dataset
- **Velicina:** 32,000 naslova
- **Balansiranost:** 16,001 nije clickbait / 15,999 clickbait — savrseno balansiran
- **Format:** CSV sa kolonama `headline` (tekst) i `clickbait` (0 ili 1)
- **Podela:**
  - Train: 28,800 naslova (90%)
  - Validacija: 2,560 naslova (8%)
  - Test: 640 naslova (2%)

**Vazno:** Podela je fiksirana unapred u zasebne fajlove — test podaci nisu vidjeni tokom treninga.

---

## 3. Koja je arhitektura modela?

**Encoder-only Transformer** — slican BERT arhitekturi, ali gradjen od nule bez pre-trained tezina.

### Zasto Encoder-only a ne pun Transformer?
Pun Transformer (Encoder + Decoder) se koristi za **generisanje teksta** (prevod, sumarizacija).
Za **klasifikaciju** treba samo Encoder — on razume tekst, nema potrebe za generisanjem.

### Tok podataka kroz model:

```
Naslov (string)
    ↓
Tokenizacija → [SOS] word1 word2 ... wordK [EOS] [PAD] ... [PAD]
    ↓
Input Embeddings → svaki token postaje vektor dimenzije 128
    ↓
Positional Encoding → dodaje informaciju o poziciji tokena u recenici
    ↓
Encoder (6 blokova):
    svaki blok:
        Multi-Head Attention (8 glava) → tokeni primaju kontekst jedni od drugih
        Feed-Forward Network          → nelinearnost, dublje ucenje
        Layer Normalization           → stabilizacija treninga
    ↓
[SOS] token output (pozicija 0) → vektor dimenzije 128
    ↓
Classification Head → Linear(128, 1)
    ↓
Sigmoid → verovatnoca [0, 1]
    ↓
>= 0.5 → CLICKBAIT / < 0.5 → NIJE CLICKBAIT
```

---

## 4. Sta je Multi-Head Attention i zasto je bitan?

Attention mehanizam omogucava svakom tokenu da "pogleda" sve ostale tokene i odluci koji su mu relevantni.

**Primer:** U recenici "You Won't Believe What Happened" — rec "Believe" je direktno povezana sa "Won't" i zajedno nose znacenje tipicno za clickbait.

**Multi-Head** znaci da se ovo radi paralelno na vise nacina (8 glava) — svaka glava uci drugaciju vrstu odnosa izmedju reci. Vise glava = model moze da razume vise razlicitih obrazaca istovremeno.

---

## 5. Sta je [SOS] token i zasto se koristi za klasifikaciju?

Na pocetku svake sekvence dodaje se specijalni **[SOS]** token (Start of Sentence).
Kroz Encoder blokove, ovaj token prima kontekst od **svih** ostalih tokena u recenici.

Na kraju Encoder-a, vektor [SOS] tokena sadrzi **sumarnu reprezentaciju cele recenice** — zato se koristi za klasifikaciju (kao [CLS] token u BERT-u).

---

## 6. Sta je Loss funkcija?

**BCEWithLogitsLoss** (Binary Cross Entropy with Logits) — standardna loss funkcija za binarnu klasifikaciju.

Meri koliko se predikcija modela razlikuje od tacnog odgovora:
- Model predvidi 0.9 (clickbait), tacno je 1 → mali loss
- Model predvidi 0.9 (clickbait), tacno je 0 → veliki loss

Tokom treninga optimizer smanjuje ovaj loss azuriranjem tezina modela.

---

## 7. Sta je tokenizator?

Tokenizator konvertuje tekst u listu brojeva (token ID-eva) koje model moze da obradjuje.

- Tip: **Word-Level** — svaka rec je jedan token
- Specijalni tokeni: `[UNK]` (nepoznata rec), `[PAD]` (punjenje), `[SOS]` (pocetak), `[EOS]` (kraj)
- Gradjen **samo na trening podacima** — da se izbegne data leakage
- Velicina recnika: ~8,000-13,000 reci

**Padding:** Svi naslovi se dopunjuju do fiksne duzine od 64 tokena. Krace se dopunjavaju [PAD] tokenima, duze se skracuju.

---

## 8. Kako se trenira model?

1. Naslov prolazi kroz model i dobija se logit (broj)
2. Sigmoid pretvara logit u verovatnocu
3. Loss funkcija poredi verovatnocu sa tacnim odgovorom
4. Backpropagation racuna gradijente
5. Adam optimizer azurira tezine modela
6. Ponavlja se za svaki batch (64 naslova) kroz sve epohe

**Hiperparametri:**
- Learning rate: 0.0003
- Batch size: 64
- Epohe: 100
- Model dimension: 128
- Encoder blokova: 6
- Attention glava: 8

---

## 9. Kako se evaluira model?

### Metrike:
- **Accuracy** — % ispravno klasifikovanih naslova
- **Precision** — od svih koje je model rekao da su clickbait, koliko zaista jesu
- **Recall** — od svih clickbait naslova, koliko ih je model pronasao
- **F1 Score** — harmonijska sredina Precision i Recall — glavna metrika

### Zasto F1 a ne samo Accuracy?
Dataset je balansiran pa su skoro iste, ali F1 bolje detektuje ako model preferira jednu klasu.

### Postignuti rezultati:
- Epoha 0: Accuracy **98.55%**, F1 **98.58%**
- Epoha 1: Accuracy **98.52%**, F1 **98.54%**

---

## 10. Zasto ne koristis pre-trained model kao BERT?

Cilj projekta je razumevanje i implementacija Transformer arhitekture od nule.
BERT bi dao slicne ili bolje rezultate, ali ne bi demonstrirao razumevanje arhitekture.
Trening od nule pokazuje da model moze da nauci zadatak bez ikakvog prethodnog znanja o jeziku.

---

## 11. Moguca pitanja profesora

**P: Zasto encoder-only a ne BERT direktno?**
O: Gradim od nule da demonstriram razumevanje arhitekture. BERT je pre-trained encoder, moj model je ista arhitektura trenirana specificno na ovom datasetu.

**P: Zasto [SOS] token za klasifikaciju?**
O: [SOS] token prima attention od svih tokena u recenici kroz sve encoder blokove, pa na kraju sadrzi sumarnu reprezentaciju cele recenice. Ovo je standardan pristup — BERT koristi isti princip sa [CLS] tokenom.

**P: Kako si osigurao da test podaci nisu vidjeni tokom treninga?**
O: Dataset je podijeljen unapred u zasebne fajlove (train/val/test) sa fiksiranim seed-om. Tokenizator je gradjen iskljucivo na train skupu.

**P: Sta bi poboljsalo model?**
O: Veci model_dimension (512 kao BERT), vise epoha sa early stopping, data augmentation, ili fine-tuning pre-trained modela.

**P: Zasto je loss tako nizak vec posle prve epohe?**
O: Dataset je relativno jednostavan za transformer — naslovi su kratki i clickbait obrasci su jasni (broj lista, direktno obracanje, senzacionalne reci). Model brzo prepoznaje ove obrasce.
