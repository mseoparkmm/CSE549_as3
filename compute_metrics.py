# recompute_metrics.py
# Usage:
#   python recompute_metrics.py --file ai_team_results.csv

import json
import argparse
import pandas as pd

ENCODINGS = ("utf-8", "utf-8-sig", "latin1")

def load_logs(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise SystemExit(f"[error] failed to read CSV with encodings {ENCODINGS}: {last_err}")

def parse_letters(s: str):
    try:
        return json.loads(s)
    except Exception:
        # tolerate simple-quote JSON etc.
        try:
            return json.loads(str(s).replace("'", '"'))
        except Exception:
            return []

def compute_metrics(df: pd.DataFrame):
    # required columns
    needed = {"ground_truth_letter", "elim_letters"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] missing columns in CSV: {missing}")

    total = int(len(df))
    if total == 0:
        return {"trials": 0, "decider_correct": 0, "decider_accuracy": 0.0,
                "assistant_correct": 0, "assistant_accuracy": 0.0}

    # --- Decider accuracy (final answer correct) ---
    if "is_correct" in df.columns:
        decider_correct = int(df["is_correct"].sum())
    elif "decider_choice" in df.columns:
        decider_correct = int(
            (df["decider_choice"].astype(str).str.upper() ==
             df["ground_truth_letter"].astype(str).str.upper()).sum()
        )
    else:
        raise SystemExit("[error] neither 'is_correct' nor 'decider_choice' present")
    decider_acc = decider_correct / total

    # --- Assistant accuracy (did NOT eliminate the correct answer) ---
    def correct_removed(row) -> bool:
        elim_letters = parse_letters(row["elim_letters"])
        corr = str(row["ground_truth_letter"]).strip().upper()
        return corr in [str(x).strip().upper() for x in elim_letters]

    removed_mask = df.apply(correct_removed, axis=1)
    assistant_correct = int((~removed_mask).sum())  # kept the correct answer
    assistant_acc = assistant_correct / total

    return {
        "trials": total,
        "decider_correct": decider_correct,
        "decider_accuracy": decider_acc,
        "assistant_correct": assistant_correct,
        "assistant_accuracy": assistant_acc
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="ai_team_results.csv")
    args = ap.parse_args()

    df = load_logs(args.file)
    m = compute_metrics(df)

    print("=== Recomputed Metrics ===")
    print(f"Trials:                 {m['trials']}")
    print(f"Decider's accuracy:        {m['decider_correct']} / {m['trials']}  ({m['decider_accuracy']*100:.1f}%)")
    print(f"Assistant's accuracy : {m['assistant_correct']} / {m['trials']}  ({m['assistant_accuracy']*100:.1f}%)")

if __name__ == "__main__":
    main()
