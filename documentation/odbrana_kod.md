# Implementacija — Objašnjenje koda

---

## 1. Konfiguracija (`config.py`)

```python
def get_config():
    return {
        "batch_size": 64,        # broj primera po batch-u tokom treninga
        "num_epochs": 100,       # ukupan broj epoha treninga
        "learning_rate": 3e-4,   # stopa učenja za Adam optimizer
        "context_size": 64,      # maksimalna dužina naslova u tokenima
        "model_dimension": 128,  # dimenzija vektora embeddings-a
        "model_folder": "weights",
        "model_basename": "clickbait_detector_",
        "preload": None,         # 'latest' za nastavak treninga, None za novi start
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/clickbait_detection",  # TensorBoard logovi
        "seed": 561,             # seed za reproduktivnost
        "train_file": "data/train.csv",
        "val_file":   "data/val.csv",
        "test_file":  "data/test.csv"
    }
```

Konfiguracija centralizeuje sve hiperparametre modela na jednom mestu.
Vrednost `preload: None` znači trening kreće od početka; vrednost `'latest'`
omogućava nastavak prekinutog treninga učitavanjem poslednje sačuvanih weights-a.

---

## 2. Dataset i tokenizacija (`dataset.py`, `train.py`)

### Tokenizator

```python
def get_or_build_tokenizer(config, dataset, min_frequency=2):
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = CharDelimiterSplit(' ')  # split po razmaku

    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=min_frequency  # reči koje se pojavljuju manje od 2x → [UNK]
    )

    tokenizer.train_from_iterator(get_all_sentences(dataset), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
```

Koristi se **WordLevel** tokenizator — svaka reč dobija jedinstveni celobrojni ID.
Reči koje se pojavljuju samo jednom u trening skupu mapiraju se na `[UNK]` token.
Tokenizator se trenira isključivo na trening skupu kako bi se izbeglo **curenje podataka** (data leakage).

Specijalni tokeni:
- `[UNK]` — nepoznata reč
- `[PAD]` — dopuna do fiksne dužine
- `[SOS]` — oznaka početka sekvence; koristi se kao klasifikacioni token (ekvivalent `[CLS]` u BERT-u)
- `[EOS]` — oznaka kraja sekvence

### Dataset klasa

```python
class ClickbaitDataset(TorchDataset):
    def __getitem__(self, index):
        headline = str(row['headline']).lower()  # normalizacija na mala slova
        label = float(row['clickbait'])           # 0.0 ili 1.0

        # Tokenizacija naslova
        input_tokens = self.tokenizer.encode(headline).ids
        input_tokens = input_tokens[:self.context_size - 2]  # skraćivanje ako je predugačak

        # Dopuna padding tokenima do fiksne dužine
        num_padding = self.context_size - len(input_tokens) - 2

        # Finalni encoder input: [SOS] token1 token2 ... tokenN [EOS] [PAD]...[PAD]
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * num_padding, dtype=torch.int64)
        ], dim=0)

        # Maska skriva [PAD] tokene od attention mehanizma
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()

        return {
            "encoder_input": encoder_input,   # shape: (context_size,)
            "encoder_mask":  encoder_mask,    # shape: (1, 1, context_size)
            "label":         torch.tensor(label, dtype=torch.float32),
            "headline":      headline
        }
```

Svaki naslov se pretvara u sekvencu fiksne dužine `context_size = 64`.
`[SOS]` token uvek stoji na poziciji 0 — njegov izlaz iz enkodera koristi se za klasifikaciju.
Maska osigurava da `[PAD]` tokeni ne utiču na attention težine.

---

## 3. Arhitektura modela (`model.py`)

### Input Embeddings

```python
class InputEmbeddings(nn.Module):
    def forward(self, x):
        # Množenje sa sqrt(model_dimension) — skaliranje po "Attention is All You Need"
        return self.embedding(x) * math.sqrt(self.model_dimension)
```

Svaki token ID mapira se u vektor dimenzije `model_dimension = 128`.
Skaliranje sa `√d_model` sprečava da početne vrednosti budu premale u odnosu na pozicione enkodinge.

### Poziciono Enkodiranje

```python
class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, context_size, dropout):
        # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        positional_encodings[:, 0::2] = torch.sin(position * div_term)  # parni indeksi
        positional_encodings[:, 1::2] = torch.cos(position * div_term)  # neparni indeksi

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]  # dodaje pozicionu informaciju
        return self.dropout(x)
```

Transformer nema rekurentnost niti konvoluciju, pa poziciono enkodiranje daje modelu
informaciju o redosledu tokena u sekvenci. Koriste se sinusne i kosinusne funkcije
različitih frekvencija — isti obrazac iz originalnog rada "Attention is All You Need" (Vaswani et al., 2017).

### Layer Normalizacija

```python
class LayerNormalization(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        # alpha i bias su naučeni parametri
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
```

Normalizuje aktivacije unutar svakog sloja, što stabilizuje trening i
ubrzava konvergenciju. `eps = 1e-6` sprečava deljenje nulom.

### Multi-Head Attention

```python
class MultiHeadAttentionBlock(nn.Module):
    @staticmethod
    def attention(query, key, value, mask, dropout):
        # Attention(Q, K, V) = softmax(QK^T / √d_k) · V
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dimension)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)  # maskira PAD tokene

        attention_scores = attention_scores.softmax(dim=-1)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # linearna projekcija Q
        key   = self.w_k(k)  # linearna projekcija K
        value = self.w_v(v)  # linearna projekcija V

        # Podela na H glava: (batch, context, d_model) → (batch, H, context, d_k)
        query = query.view(..., self.heads, self.head_dimension).transpose(1, 2)
        key   = key.view(...,   self.heads, self.head_dimension).transpose(1, 2)
        value = value.view(..., self.heads, self.head_dimension).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Spajanje glava: (batch, H, context, d_k) → (batch, context, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.head_dimension)

        return self.w_o(x)  # izlazna linearna projekcija
```

Multi-head attention paralelno izračunava `H = 8` attention funkcija.
Svaka glava nauči da obraća pažnju na različite aspekte naslova
(npr. jedna glava na senzacionalističke reči, druga na strukturu rečenice).
Deljenje sa `√d_k` sprečava da softmax uđe u zasićenu zonu za veće dimenzije.

### Feed-Forward Blok

```python
class FeedForwardBlock(nn.Module):
    def forward(self, x):
        # FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
```

Svaki token se obrađuje nezavisno kroz dvoslojnu neuronsku mrežu.
Unutrašnja dimenzija `feed_forward_dimension = 512` (4× model_dim) daje modelu
kapacitet za složenije transformacije koje attention nije uhvatio.

### Rezidualna veza (Residual Connection)

```python
class ResidualConnection(nn.Module):
    def forward(self, x, sublayer):
        # Pre-norm varijanta: x + dropout(sublayer(LayerNorm(x)))
        return x + self.dropout(sublayer(self.norm(x)))
```

Rezidualna veza (skip connection) rešava problem nestajanja gradijenata u dubokim mrežama.
Implementirana je `pre-norm` varijanta — normalizacija se primenjuje pre podsloja,
što je stabilnije za trening od originalne `post-norm` varijante.

### Encoder blok i Encoder

```python
class EncoderBlock(nn.Module):
    def forward(self, x, source_mask):
        # 1. Self-attention sa rezidualnom vezom
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))
        # 2. Feed-forward sa rezidualnom vezom
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def forward(self, x, mask):
        for layer in self.layers:  # prolazi kroz svih N=6 blokova
            x = layer(x, mask)
        return self.norm(x)        # finalna normalizacija
```

Svaki od 6 encoder blokova primenjuje:
1. Self-attention (svaki token gleda sve ostale tokene)
2. Feed-forward mrežu

Obe operacije imaju rezidualnu vezu i layer normalizaciju.

### Klasifikaciona glava

```python
class ClassificationHead(nn.Module):
    def forward(self, x):
        # x: (batch, context_size, model_dimension)
        cls_output = x[:, 0, :]       # uzima [SOS] token sa pozicije 0 → (batch, model_dim)
        return self.proj(cls_output).squeeze(-1)  # linearna projekcija → (batch,) logit
```

Nakon što svih 6 encoder blokova obrade sekvencu, uzima se **reprezentacija [SOS] tokena**
sa pozicije 0. Ovaj vektor dimenzije 128 sadrži kontekstualizovanu informaciju o celom naslovu.
Linearna projekcija `Linear(128, 1)` svodi ga na jedan logit koji ide u `BCEWithLogitsLoss`.

### Inicijalizacija parametara

```python
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)  # Xavier inicijalizacija za 2D+ tenzore
```

Xavier uniform inicijalizacija osigurava da varijansa aktivacija ostane stabilna
kroz sve slojeve na početku treninga.

---

## 4. Trening (`train.py`)

```python
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Optimizer: Adam sa learning rate 3e-4 i eps=1e-9
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    # Funkcija gubitka: BCEWithLogitsLoss (numerički stabilnija od BCE + Sigmoid)
    loss_function = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        for batch in train_dataloader:

            encoder_input = batch['encoder_input'].to(device)  # (batch, context_size)
            encoder_mask  = batch['encoder_mask'].to(device)   # (batch, 1, 1, context_size)
            label         = batch['label'].to(device)          # (batch,)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)  # (batch, context, d_model)
            logits = model.classify(encoder_output)                      # (batch,)

            loss = loss_function(logits, label)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validacija na kraju svake epohe
        run_validation(model, val_dataloader, ...)

        # Čuvanje weights-a na epohi 0, svakih 10 epoha i na kraju
        if epoch % 10 == 9 or epoch == 0 or epoch == config['num_epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)
```

Trening prati standardni PyTorch ciklus:
1. **Forward pass** — ulaz prolazi kroz model i dobija se logit
2. **Izračunavanje gubitka** — `BCEWithLogitsLoss` poredi logit sa labelom (0 ili 1)
3. **Backward pass** — gradijenti se propagiraju unazad kroz mrežu
4. **Ažuriranje parametara** — Adam optimizer pomera parametre u smeru smanjenja gubitka

`BCEWithLogitsLoss` kombinuje sigmoid i binary cross-entropy u jednoj operaciji što je numerički stabilnije.

---

## 5. Evaluacija (`test.py`)

```python
def run_test(model, test_dataloader, device):
    model.eval()  # isključuje dropout za evaluaciju

    tp = fp = tn = fn = 0  # true/false positives/negatives

    with torch.no_grad():  # ne izračunava gradijente — brže i manje memorije
        for batch in test_dataloader:
            encoder_output = model.encode(encoder_input, encoder_mask)
            logits = model.classify(encoder_output)

            # Sigmoid konvertuje logit u verovatnoću, prag 0.5 → binarna odluka
            predicted = (torch.sigmoid(logits) >= 0.5).float()

            tp += ((predicted == 1) & (label == 1)).sum().item()
            fp += ((predicted == 1) & (label == 0)).sum().item()
            tn += ((predicted == 0) & (label == 0)).sum().item()
            fn += ((predicted == 0) & (label == 1)).sum().item()

    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    f1        = 2 * precision * recall / (precision + recall)
```

Metrike:
- **Accuracy** — udeo tačnih predikcija
- **Precision** — od svih koje je model označio kao clickbait, koliko zaista jeste
- **Recall** — od svih pravih clickbait naslova, koliko je model pronašao
- **F1 score** — harmonijska sredina precision-a i recall-a; balansira obe greške

---

## 6. Klasifikacija novog naslova (`test.py → classify_headline`)

```python
def classify_headline(headline: str) -> str:
    # Tokenizacija na isti način kao tokom treninga
    tokens = tokenizer.encode(headline.lower()).ids
    tokens = tokens[:context_size - 2]
    num_padding = context_size - len(tokens) - 2

    encoder_input = torch.tensor(
        [sos_id] + tokens + [eos_id] + [pad_id] * num_padding
    ).unsqueeze(0)  # dodaje batch dimenziju: (1, context_size)

    encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int()

    with torch.no_grad():
        encoder_output = model.encode(encoder_input, encoder_mask)
        logit = model.classify(encoder_output)
        prob = torch.sigmoid(logit).item()  # verovatnoća da je clickbait [0, 1]

    label = "CLICKBAIT" if prob >= 0.5 else "NOT CLICKBAIT"
    return f"{label} (confidence: {prob:.2%})"
```

Novi naslov prolazi kroz isti preprocessing pipeline kao tokom treninga:
mala slova → tokenizacija → truncation → padding → maska.
Sigmoid funkcija pretvara logit u verovatnoću u opsegu [0, 1].

---

## Pregled toka podataka

```
Naslov (string)
    ↓  lowercase + tokenizacija (WordLevel)
[SOS] [tok1] [tok2] ... [tokN] [EOS] [PAD] ... [PAD]   ← sekvenca dužine 64
    ↓  InputEmbeddings (×√128)
Matrica embeddings-a  (64 × 128)
    ↓  PositionalEncoding
Embeddings + pozicione informacije  (64 × 128)
    ↓  Encoder blok × 6 (Self-Attention + Feed-Forward + Residual)
Kontekstualizovani vektori  (64 × 128)
    ↓  ClassificationHead — uzima [SOS] na poziciji 0
Vektor reprezentacije naslova  (128,)
    ↓  Linear(128 → 1) + Sigmoid
Verovatnoća clickbait-a  [0.0 – 1.0]
```
