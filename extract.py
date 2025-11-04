import re
import json
import requests
import pandas as pd

URL = "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt"
OUT_FULL = "logiqa_eval.csv"
OUT_200  = "logiqa_eval_200.csv"
sample_200 = False 

OPTION_PREFIX_RE = re.compile(r"^[A-D]\s*[\.\:)\]．、]\s*")

def clean_option_prefix(s: str) -> str:
    return OPTION_PREFIX_RE.sub("", s, count=1).strip()

def load_text(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw = resp.content
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")

def normalize_lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.split("\n")

def parse_eval(lines: list[str]) -> list[dict]:
    examples = []
    n = len(lines) // 8
    bad = 0
    for k in range(n):
        row = 8 * k
        if row + 7 >= len(lines):
            break

        ans_raw = (lines[row + 1] or "").strip()
        context = (lines[row + 2] or "").strip()
        question = (lines[row + 3] or "").strip()
        opts_raw = [(lines[row + 4 + i] or "").strip() for i in range(4)]

        correct_letter = ans_raw.replace(".", "").strip().upper()
        if correct_letter not in ("A", "B", "C", "D"):
            bad += 1
            continue

        options = [clean_option_prefix(o) for o in opts_raw]

        letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        ex = {
            "context": context,
            "question": question,
            "choices": json.dumps(options, ensure_ascii=False),
            "correct_answer": letter2idx[correct_letter],
        }
        examples.append(ex)

    if bad:
        print(f"[warn] skipped malformed items: {bad}")
    return examples

def main():
    print("[load] downloading Eval.txt ...")
    text = load_text(URL)
    lines = normalize_lines(text)
    print(f"[info] total raw lines: {len(lines)}")

    print("[parse] building examples in 8-line blocks ...")
    examples = parse_eval(lines)
    print(f"[info] parsed items: {len(examples)} (expected ~651)")

    df = pd.DataFrame(examples, columns=["context", "question", "choices", "correct_answer"])
    df.to_csv(OUT_FULL, index=False, encoding="utf-8")
    print(f"[save] {OUT_FULL}  rows={len(df)}")

    if sample_200:
        if len(df) >= 200:
            df.sample(200).to_csv(OUT_200, index=False, encoding="utf-8")
            print(f"[save] {OUT_200}  rows=200")
        else:
            print("[warn] not enough rows to sample 200.")

if __name__ == "__main__":
    main()