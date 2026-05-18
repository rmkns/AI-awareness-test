# Lithuanian LLM situational uncertainty evaluation

This repository contains the source code and test set used for the bachelor thesis:

**Lokaliuju dideliu kalbos modeliu situacinio suvokimo daugiakriterinis vertinimas**

The project evaluates how large language models respond to incomplete,
contradictory, ambiguous, or risk-related Lithuanian prompts. In the thesis this is
treated as a narrow operational form of situational awareness.

## Structure

- `project_a/` - Lithuanian test set and local model runner.
- `project_b_judge/` - LLM-as-Judge rubric, OpenRouter evaluation scripts, and agreement analysis.
- `project_c_intrinsic/` - intrinsic uncertainty evaluation using HuggingFace models and token/logit metrics.
- `scripts/` - convenience Windows BAT launchers and model-list files used in the final experiments.

## What is intentionally not included

- API keys and `_apikey.bat`
- virtual environments
- HuggingFace caches or model weights
- generated `runs/` outputs
- Python `__pycache__`
- large temporary result files

## Basic setup

Use Python 3.12+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Project B needs an OpenRouter key:

```powershell
set "OPENROUTER_API_KEY=sk-or-..."
```

Project C may need a HuggingFace token for gated models:

```powershell
set "HF_TOKEN=hf_..."
```

## Running the parts

Project A:

```powershell
cd project_a
python run_tests.py --help
```

Project B:

```powershell
cd project_b_judge
python evaluate_llm_judge_resilient.py --help
python analyze_agreement.py --help
```

Project C:

```powershell
cd project_c_intrinsic
python check_hf_models.py --help
python intrinsic_evaluator_resilient.py --help
python entropy_analyzer.py --help
python correlate_methods.py --help
```

The BAT files in `scripts/` are convenience launchers from the local experiment
environment. Before reuse on another machine, check paths and environment
variables in those files.

