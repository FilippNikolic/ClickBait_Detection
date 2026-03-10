from test import classify_headline

# Ucitaj naslove iz fajla (ignorisi komentare i prazne linije)
with open("data/sample_headlines.txt", "r", encoding="utf-8") as f:
    headlines = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

print("=" * 60)
print("CLICKBAIT DETEKCIJA")
print("=" * 60)

for headline in headlines:
    result = classify_headline(headline)
    print(f"{result}")
    print(f"  -> {headline}")
    print()
