# Project C: Intrinsic uncertainty measurement

Trečiasis bakalaurinio projektas. Matuoja modelio neapibrėžtumą
**iš logito lygmens**, prieinant tiesiai prie HuggingFace transformer
svorių. Tai trečiasis trianguliacijos sluoksnis šalia raktažodžių
detekcijos (A) ir LLM-as-Judge (B).

## Failai

| Failas | Paskirtis |
|---|---|
| `intrinsic_evaluator.py` | Generavimas su per-token logito kaupimu (Shannon H, top-k margin) |
| `entropy_analyzer.py` | Atsakymo lygmens metrikų agregavimas, kategorijų sumarizavimas |
| `correlate_methods.py` | Trianguliacija — Pearson r / Spearman ρ tarp A, B, C |
| `requirements.txt` | Python paketai |

## Setup

```bash
cd project_c_intrinsic
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Reikalavimai:** GPU su ≥ 8 GB VRAM rekomenduojamas.
> Llama-3.2-3B veikia ant 8 GB su fp16. Qwen2.5-7B fp16 reikalauja
> ≥ 16 GB, bet su `--load-in-4bit` (bitsandbytes nf4) telpa į ~5 GB,
> su `--load-in-8bit` — į ~8 GB. Testuota ant RTX 5070 12 GB.
> Naudojant `--device cpu` veikia, bet labai lėtai (5+ min/atsakymą).
>
> **RTX 50 serija (Blackwell, sm_120):** reikia PyTorch su CUDA 12.8+
> wheels: `pip install --index-url https://download.pytorch.org/whl/cu128 torch`.
> Stock cu121 wheels mes „no kernel image" klaidą.

## Naudojimas

### 1. Surinkti intrinsic duomenis

```bash
# Llama-3.2-3B fp16 (telpa į 8+ GB VRAM)
python intrinsic_evaluator.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --tests-dir ../project_a/tests \
    --output intrinsic_out/llama32_3b.jsonl \
    --runs 5 \
    --dtype fp16

# Qwen2.5-7B 4-bit (telpa į ~5 GB VRAM, RTX 5070 12 GB tinka)
python intrinsic_evaluator.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tests-dir ../project_a/tests \
    --output intrinsic_out/qwen25_7b.jsonl \
    --runs 5 \
    --load-in-4bit
```

Skriptas:
- įkelia modelį į GPU/CPU,
- kiekvienam testui (40 vnt.) paleidžia 5 generavimus,
- per-token Shannon entropija ir tikimybės dispersija saugoma,
- atsakymo lygmens metrikos (mean H, max H, low_conf_ratio) rašomos
  į JSONL.

> **Kvantizavimo akademinė pastaba:** 4-bit/8-bit svorių kvantizavimas
> šiek tiek keičia logito pasiskirstymą (ypač žemo dažnio token'ams),
> todėl absoliučios entropijos reikšmės nelyginamos tarp fp16 ir
> kvantuoto modelio. Tai būtina paminėti tezės metodologijos
> skyriuje. Vidaus rangavimas (kuri kategorija turi aukščiausią
> entropiją) ir trianguliacijos koreliacija su A/B sluoksniais
> išlieka patikima.

### 2. Agreguoti / vizualizuoti

```bash
python entropy_analyzer.py \
    --intrinsic intrinsic_out/llama32_3b.jsonl \
    --output-dir analysis/intrinsic_llama32
```

Sukuria `per_category_summary.csv` — kuri kategorija sukelia aukščiausią
entropiją? Lauktinas rezultatas: kategorija `unknown_*` turi aukščiausią
mean H.

### 3. Trianguliacija (palyginimas su A ir B)

```bash
python correlate_methods.py \
    --intrinsic intrinsic_out/llama32_3b.jsonl \
    --judged ../project_b_judge/judged/llama32_gpt4o.jsonl \
    --output-dir analysis/triangulation_llama32
```

Apskaičiuoja koreliaciją tarp:
- `mean_entropy` ↔ `keyword UNK` signalo,
- `max_entropy` ↔ `keyword CONTR`,
- `low_conf_ratio` ↔ `judge CLARIFY` ir t.t.

Sukuria scatter plot'us `*.png` formatu.

## Output JSONL schema (intrinsic)

```json
{
  "model": "meta-llama/Llama-3.2-3B-Instruct",
  "test_id": "09_unknown_future_budget",
  "run": 0,
  "response": "...",
  "n_tokens": 187,
  "mean_entropy": 1.842,
  "max_entropy": 7.231,
  "min_p_chosen": 0.043,
  "low_conf_count": 23,
  "low_conf_ratio": 0.123,
  "elapsed_s": 4.21
}
```

Su `--keep-full-tokens` papildomai įrašomas `tokens` masyvas su
per-token statistika (didelis failas — 80 KB/atsakymas).

## Modelių pasirinkimas

Reikia HuggingFace svorių — ne visi projekto~A modeliai prieinami:

| Modelis | HF prieinamumas | Pastabos |
|---|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | ✅ (po sutikimo) | Lengvas, tinka greitai iteracijai |
| `Qwen/Qwen2.5-7B-Instruct` | ✅ | Geras LT gebėjimas |
| `google/gemma-2-2b-it` | ✅ (po sutikimo) | Pakeičia Gemma 3 |
| GPT-OSS | ⚠️ (tik specifinės versijos) | Tikrinti dabartinę licenciją |
| DeepSeek-R1 | ⚠️ (distill-* yra atviri) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |

Rekomenduojama paleisti su 2 modeliais:
1. **Llama-3.2-3B** — atstovauja mažam parametrų klasei
2. **Qwen2.5-7B** — atstovauja vidutiniam dydžiui

## Žinomi apribojimai

- **Greedy decoding** maskuoja sample-time įvairovę. Galima
  pakeisti su `--temperature 0.7` jeigu norima sample-driven analizės.
- **Token-lygmens entropija** nėra tas pats kas atsakymo
  lygmens *epistemic uncertainty* — tai paviršutinis approxim.
- **Lokali decompozicija** — UNK frazė gali sukelti **žemos
  entropijos** zoną (modelis užtikrintai pasirenka „nežinau"),
  o ne aukštos. Tai svarbi nuance interpretuojant koreliaciją.
- **Modelio dydžio efektas** — mažesni modeliai turi aukštesnę
  bazinę entropiją (didesnis žinių neapibrėžtumas), todėl
  kategorijų lyginimas viduje vieno modelio yra patikimesnis nei
  tarp modelių.

## Praktinis patarimas

Pradėk su mažu setu:
```bash
python intrinsic_evaluator.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --tests-dir ../project_a/tests \
    --output intrinsic_out/test.jsonl \
    --runs 1 \
    --max-tests 5
```

Patikrink, ar modelis užkraunamas, ar generavimas veikia, ir tik
tada paleisk full run.
