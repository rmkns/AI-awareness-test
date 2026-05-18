# Project B: LLM-as-Judge awareness evaluation

Antrasis bakalaurinio projektas. Pakartotinai įvertina projekto~A
surinktus atsakymus naudojant LLM-kaip-vertintojo (LLM-as-Judge)
metodologiją. Po to lygina su raktažodžių detekcijos rezultatais
per Cohen κ statistiką.

## Failai

| Failas | Paskirtis |
|---|---|
| `rubric.py` | 5 indikatorių rubric'as (0-2 skalė), prompt builder, JSON parser |
| `evaluate_llm_judge.py` | Pagrindinis vertinimo pipeline (paralelinis, resume-friendly) |
| `analyze_agreement.py` | Cohen κ, konfūzijos matricos, disagreements |
| `multi_judge_agreement.py` | Fleiss κ tarp 3 skirtingų judge modelių |
| `requirements.txt` | Python paketai |

## Setup

```bash
cd project_b_judge
python -m venv .venv
source .venv/bin/activate          # arba .venv\Scripts\activate Windows
pip install -r requirements.txt

export OPENROUTER_API_KEY="sk-or-..."
```

> Naudojama OpenAI-suderinama API. Numatyta — OpenRouter (vienintelis API
> visiems modeliams), bet veikia ir su tiesiogine OpenAI / Anthropic
> API, ir su lokaliu Ollama (per `--api-base http://localhost:11434/v1`).

## Naudojimas

### 1. Vertinti vienu judge modeliu

```bash
python evaluate_llm_judge.py \
    --input-dir ../project_a/results/session_qwen3_xxx \
    --output    judged/qwen3_gpt4o.jsonl \
    --judge-model openai/gpt-4o-mini
```

Skriptas:
- nuskaito visus `*.json` iš `--input-dir`,
- patikrina, kurie atsakymai jau yra `--output` JSONL (resume),
- siunčia likusius į judge LLM,
- inkrementiškai rašo rezultatą (saugu prieš pertraukimą).

### 2. Analizuoti sutarimą su raktažodžių detektorium

```bash
python analyze_agreement.py \
    --judged judged/qwen3_gpt4o.jsonl \
    --output-dir analysis/qwen3_gpt4o
```

Sukuria:
- `merged_scores.csv` — visos eilutės su keyword + judge balais
- `agreement_overall.csv` — pool'inta Cohen κ
- `agreement_per_model.csv` — Cohen κ per modelio × indikatoriaus derinį
- `disagreements_sample.csv` — atsitiktinai parinkti nesutarimo atvejai
- `summary.md` — apibendrinanti ataskaita Markdown formatu

### 3. Multi-judge stabilumo eksperimentas

Paleisk `evaluate_llm_judge.py` tris kartus su skirtingais judge
modeliais (į skirtingus output failus), tada:

```bash
python multi_judge_agreement.py \
    --judged-files judged/qwen3_gpt4o.jsonl \
                   judged/qwen3_claude.jsonl \
                   judged/qwen3_llama70b.jsonl \
    --judge-labels gpt4o claude llama70b \
    --output-dir analysis/multijudge_qwen3
```

Apskaičiuoja Fleiss κ tarp trijų judge'ų per kiekvieną indikatorių.

## Judge modelių rekomendacijos

| Modelis | Stiprybės | Trūkumai | API kainos (~) |
|---|---|---|---|
| `openai/gpt-4o-mini` | greitas, pigus, geras LT | dažnesnis JSON nukrypimas | $0.15 / 1M tokens |
| `anthropic/claude-3.5-sonnet` | tikslus, gerai laikosi rubric'o | brangesnis | $3.00 / 1M tokens |
| `meta-llama/llama-3.3-70b-instruct` | atviro kodo, alternatyva | silpnesnė LT | $0.60 / 1M tokens |

**Patarimas:** pradžioje paleisk su `--max-items 50` ir `gpt-4o-mini`,
patikrink rubric'o derinimą; tada full run.

## Output JSONL schema

```json
{
  "test_id": "09_unknown_future_budget",
  "model": "qwen3",
  "run": 0,
  "test_prompt": "...",
  "model_response": "...",
  "keyword_scores": {"UNK": 1, "CONTR": 0, "REFL": 1, "CLARIFY": 0, "URG": 0},
  "judge_model": "openai/gpt-4o-mini",
  "judge_raw": "{...full LLM response...}",
  "judge_scores": {"UNK": 2, "CONTR": 0, "REFL": 1, "CLARIFY": 1, "URG": 0,
                   "rationale": "Aiškus nežinojimo pripažinimas..."}
}
```

## Žinomi apribojimai

- Judge taip pat yra LLM → galimas \"collusion bias\" (visi judge'ai
  klysta panašiai).
- Lietuvių kalba yra mažumos kalba LLM'uose; rezultatų stabilumas
  mažesnis nei anglų kalba.
- Rubric'as yra autoriaus suformuluotas — galima objektyvumo riba.
