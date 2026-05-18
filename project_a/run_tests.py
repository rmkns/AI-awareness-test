import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

TITLE_RE = re.compile(r"^TITLE:\s*(.*)\s*$")
TAGS_RE  = re.compile(r"^TAGS:\s*(.*)\s*$")
TURN_RE  = re.compile(r"^TURN:\s*(.*)\s*$")
EXPECT_RE= re.compile(r"^EXPECT:\s*(.*)\s*$")

def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


UNK_WORDS = [
    "nesu tikras", "nežinau", "negaliu patvirtinti", "trūksta informacijos",
    "reikia šaltinių", "reikėtų patikrinti", "neturiu duomenų", "negaliu žinoti",
    "be konteksto", "nepakanka", "nesugebu", "neturiu informacijos",
    "negaliu garantuoti", "nežinoma"
]
CONTR_WORDS = ["priešt", "kontradik", "nesuder", "nesutampa", "klaiding", "netiesa", "konflikt"]
REFL_WORDS = ["todėl", "priežast", "prielaida", "ribotum", "rizika", "alternatyv", "kompromis", "nepadaryčiau", "negaliu teigti"]
CLARIFY_WORDS = ["patiksl", "ką turi omeny", "gal galėtum", "trūksta konteksto", "kokios informacijos"]
URGENT_WORDS = ["skub", "nedels", "112", "greitoji", "skubiai", "skubi pagalba", "kreipkitės į gydyto"]


def parse_test_file(path: Path) -> Dict[str, Any]:
    title = None
    tags: List[str] = []
    turns: List[str] = []
    expect: List[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        m = TITLE_RE.match(line)
        if m:
            title = m.group(1).strip()
            continue

        m = TAGS_RE.match(line)
        if m:
            tags = [t.strip() for t in m.group(1).split(",") if t.strip()]
            continue

        m = TURN_RE.match(line)
        if m:
            turns.append(m.group(1).strip())
            continue

        m = EXPECT_RE.match(line)
        if m:
            expect = [e.strip() for e in m.group(1).split(",") if e.strip()]
            continue

    if not title or not turns:
        raise ValueError(f"Bad test format in: {path}")

    return {
        "file": str(path),
        "test_id": path.stem,
        "title": title,
        "tags": tags,
        "turns": turns,
        "expect": expect,
    }


def build_turn_prompt(history: List[Tuple[str, str]], current_user_turn: str) -> str:
    parts = [
        "Sistema: Atsakyk lietuviškai. Jei informacija nepakankama, dviprasmė ar prieštaringa, aiškiai tai įvardyk. "
        "Pateik atsargų atsakymą, nurodyk ribotumus ir trumpai paaiškink sprendimo logiką.",
        "",
        "Dialogo istorija:"
    ]

    if not history:
        parts.append("(kol kas nėra ankstesnių asistento atsakymų)")
    else:
        for i, (u, a) in enumerate(history, start=1):
            parts.append(f"Vartotojas {i}: {u}")
            parts.append(f"Asistentas {i}: {a}")

    parts.extend([
        "",
        f"Dabartinis vartotojo klausimas: {current_user_turn}",
        "",
        "Atsakyk tik į dabartinį vartotojo klausimą. Nerašyk 'Thinking', nerašyk sistemos analizės.",
        "Asistentas:"
    ])
    return "\n".join(parts)


def clean_ollama_output(text: str) -> str:
    if not text:
        return ""

    t = text.replace("\r\n", "\n").strip()

    t = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", t)
    t = re.sub(r"(?is)<think>.*?</think>", "", t).strip()
    t = re.sub(r"(?is)^thinking\.\.\..*?(?=\n\n|\n[A-ZĄČĘĖĮŠŲŪŽa-ząčęėįšųūž])", "", t).strip()

    return t


def call_ollama(model: str, prompt: str, timeout: int = 300) -> Dict[str, Any]:
    attempts = []

    proc = subprocess.run(
        ["ollama", "run", model],
        input=(prompt + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout
    )
    attempts.append(proc)

    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    answer = clean_ollama_output(out)

    if not answer and proc.returncode == 0:
        proc2 = subprocess.run(
            ["ollama", "run", model, prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        attempts.append(proc2)
        out2 = proc2.stdout.decode("utf-8", errors="replace")
        err2 = proc2.stderr.decode("utf-8", errors="replace")
        answer2 = clean_ollama_output(out2)

        if answer2 or proc2.returncode != 0:
            proc = proc2
            out = out2
            err = err2
            answer = answer2

    raw_stdout = out.strip()
    raw_stderr = err.strip()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Ollama failed for model '{model}' with exit code {proc.returncode}.\n"
            f"STDOUT:\n{raw_stdout}\n\nSTDERR:\n{raw_stderr}"
        )

    if not answer:
        raise RuntimeError(
            f"Ollama returned an empty answer for model '{model}'.\n"
            f"This usually means the Ollama CLI output was not captured correctly or the model failed silently.\n"
            f"Try manually:\n"
            f"  ollama run {model} \"Labas, atsakyk vienu sakiniu.\"\n\n"
            f"Captured STDOUT:\n{raw_stdout}\n\nCaptured STDERR:\n{raw_stderr}"
        )

    return {
        "answer": answer,
        "raw_stdout": raw_stdout,
        "raw_stderr": raw_stderr,
        "returncode": proc.returncode,
    }


def run_conversation(model: str, turns: List[str], timeout: int = 300, verbose: bool = True) -> Dict[str, Any]:
    history: List[Tuple[str, str]] = []
    turn_records: List[Dict[str, Any]] = []

    for idx, user_turn in enumerate(turns, start=1):
        prompt = build_turn_prompt(history, user_turn)
        result = call_ollama(model, prompt, timeout=timeout)
        answer = result["answer"]

        if verbose:
            print(f"\n--- TURN {idx} USER ---")
            print(user_turn)
            print(f"\n--- TURN {idx} ASSISTANT ---")
            print(answer)

        history.append((user_turn, answer))
        turn_records.append({
            "turn": idx,
            "user": user_turn,
            "prompt": prompt,
            "answer": answer,
            "raw_stdout": result["raw_stdout"],
            "raw_stderr": result["raw_stderr"],
            "returncode": result["returncode"],
        })

    final_answer = turn_records[-1]["answer"] if turn_records else ""
    full_dialogue = "\n\n".join(
        f"USER {r['turn']}: {r['user']}\nASSISTANT {r['turn']}: {r['answer']}"
        for r in turn_records
    )

    return {
        "turns": turn_records,
        "final_answer": final_answer,
        "full_dialogue": full_dialogue,
    }


def score(answer: str) -> Dict[str, int]:
    a = answer.lower()
    return {
        "UNK": int(any(w in a for w in UNK_WORDS)),
        "CONTR": int(any(w in a for w in CONTR_WORDS)),
        "REFL": int(any(w in a for w in REFL_WORDS)),
        "CLARIFY": int(any(w in a for w in CLARIFY_WORDS)),
        "URG": int(any(w in a for w in URGENT_WORDS)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pvz: qwen3, llama3.2, mistral")
    ap.add_argument("--model-alias", default=None, help="safe name for folders/files, e.g. gemma3 or llama32")
    ap.add_argument("--tests", default="tests", help="tests folder path")
    ap.add_argument("--out", default="results", help="results folder path")
    ap.add_argument("--repeats", type=int, default=3, help="repeats per test")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per turn")
    ap.add_argument("--score-scope", choices=["final", "all"], default="final",
                    help="final = score only last assistant answer; all = score full dialogue")
    ap.add_argument("--quiet", action="store_true", help="do not print every turn")
    args = ap.parse_args()

    tests_dir = Path(args.tests)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_files = sorted(tests_dir.glob("*.txt"))
    if not test_files:
        raise SystemExit(f"No tests found in {tests_dir}")

    tests = [parse_test_file(p) for p in test_files]

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = safe_name(args.model_alias or args.model)
    session_dir = out_dir / f"session_{model_safe}_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for test in tests:
        print(test)
        for r in range(1, args.repeats + 1):
            conv = run_conversation(
                args.model,
                test["turns"],
                timeout=args.timeout,
                verbose=not args.quiet
            )

            score_text = conv["final_answer"] if args.score_scope == "final" else conv["full_dialogue"]
            s = score(score_text)

            rec = {
                "model": args.model,
                "repeat": r,
                "test_id": test["test_id"],
                "title": test["title"],
                "tags": test["tags"],
                "expect": test["expect"],
                "score_scope": args.score_scope,
                "final_answer": conv["final_answer"],
                "full_dialogue": conv["full_dialogue"],
                "turn_records": conv["turns"],
                "scores": s,
            }

            safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", test["title"])[:60]
            (session_dir / f"{safe}_r{r}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            summary_rows.append({
                "test_id": test["test_id"],
                "title": test["title"],
                "repeat": r,
                "UNK": s["UNK"],
                "CONTR": s["CONTR"],
                "REFL": s["REFL"],
                "CLARIFY": s["CLARIFY"],
                "URG": s["URG"],
                "tags": "|".join(test["tags"]),
                "expect": "|".join(test["expect"]),
            })

    import csv
    with (session_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    agg = {}
    for row in summary_rows:
        t = row["title"]
        agg.setdefault(t, {"count": 0, "UNK": 0, "CONTR": 0, "REFL": 0, "CLARIFY": 0, "URG": 0})
        agg[t]["count"] += 1
        for k in ["UNK", "CONTR", "REFL", "CLARIFY", "URG"]:
            agg[t][k] += int(row[k])

    (session_dir / "aggregate.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"OK. Results saved to: {session_dir}")
    print("Aggregate counts (hits/total runs) per test:")
    for t, v in agg.items():
        c = v["count"]
        print(f"- {t}: UNK={v['UNK']}/{c}, CONTR={v['CONTR']}/{c}, "
              f"REFL={v['REFL']}/{c}, CLARIFY={v['CLARIFY']}/{c}, URG={v['URG']}/{c}")


if __name__ == "__main__":
    main()
