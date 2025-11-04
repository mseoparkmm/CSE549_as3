# evaluate_ai_team.py
# ------------------------------------------
# Usage:
#   pip install pandas python-dotenv openai tqdm
#   python evaluate_ai_team.py --data logiqa_eval.csv --n 200 --verbose --checkpoint-every 25 --save-raw
#
# Outputs:
#   - ai_team_results.csv : per-item logs (final; includes letters + texts + rationales)
#   - ai_team_results_ckpt.csv : checkpoint logs (optional)
#   - Console:
#       * per-item one-line trace (letters only), e.g.
#           [3] killed=[B, D] kept=[A, C] (correct=D) -> decider=A ok=False
#       * final summary block (accuracy, removed rate, etc.)
#
import os, json, time, argparse, random
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ---------- Config ----------
ELIMS_TO_REMOVE = 2
ELIM_MODEL = "o4-mini"        # or "gpt-4o-mini"
DECIDE_MODEL = "o4-mini"      # or "gpt-4o-mini"
USE_OPENAI_DEFAULT = True     # --dry-run to disable OpenAI calls
MAX_TRIES = 2
SLEEP_BETWEEN = 0.15          # small pause to be gentle to API

LETTER = ["A", "B", "C", "D"]
def idx2letter(i): return LETTER[i]
def letter2idx(ch): return LETTER.index(ch)

# ---------- OpenAI client ----------
def get_client():
    from openai import OpenAI
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=api_key)

# ---------- Eliminator ----------
def call_openai_eliminator(client, context, question, choices, verbose=False):
    prompt = f"""
        You assist on a 4-option multiple-choice logical reasoning problem.
        Your job:
        1) Identify EXACTLY {ELIMS_TO_REMOVE} options that are VERY LIKELY INCORRECT.
        2) Provide a SHORT, CLEAR reason for each eliminated option grounded in the passage/question.
        3) Do NOT provide the final answer. Only eliminations and reasons.

        Return STRICT JSON ONLY (no extra text):
        {{
        "eliminate": ["A","B"],
        "rationale": {{
            "A": "short reason",
            "B": "short reason"
        }}
        }}

        Passage:
        {context}

        Question:
        {question}

        Options:
        A) {choices[0]}
        B) {choices[1]}
        C) {choices[2]}
        D) {choices[3]}
    """.strip()

    last_err = None
    for attempt in range(1, MAX_TRIES+1):
        try:
            resp = client.chat.completions.create(
                model=ELIM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            txt = resp.choices[0].message.content.strip()
            # strip markdown fences if present
            if txt.startswith("```"):
                txt = txt.strip("`")
                lines = txt.splitlines()
                if lines and lines[0].lower().startswith("json"):
                    txt = "\n".join(lines[1:])
            data = json.loads(txt)

            elim_letters = data.get("eliminate", [])
            rat = data.get("rationale", {})

            # normalize letters -> indices
            seen = set()
            elim_idxs = []
            for ch in elim_letters:
                up = str(ch).strip().upper()
                if up in LETTER:
                    i = letter2idx(up)
                    if i not in seen:
                        elim_idxs.append(i); seen.add(i)
            elim_idxs = elim_idxs[:ELIMS_TO_REMOVE]

            # rationales (by letter)
            rationales = {}
            for ch in elim_letters:
                up = str(ch).strip().upper()
                if up in LETTER and letter2idx(up) in elim_idxs:
                    reason = str(rat.get(up, "")).strip() or "Eliminated as implausible."
                    rationales[up] = reason

            return elim_idxs, rationales, txt, None
        except Exception as e:
            last_err = str(e)
            if verbose:
                print(f"[warn][Eliminator attempt {attempt}] {last_err}")
            time.sleep(SLEEP_BETWEEN)

    # fallback: random eliminations
    rng = random.SystemRandom()
    idxs = list(range(4)); rng.shuffle(idxs)
    elim = idxs[:ELIMS_TO_REMOVE]
    rats = {idx2letter(i): "Eliminated (fallback random)" for i in elim}
    return elim, rats, "{}", last_err or "fallback-random"

# ---------- Decider ----------
def call_openai_decider(client, context, question, kept_pairs, verbose=False):
    (L1, t1), (L2, t2) = kept_pairs
    prompt = f"""
You must choose the single best answer between two options for a logical reasoning question.
Provide ONLY the chosen letter ("{L1}" or "{L2}") as raw text. No explanation.

Passage:
{context}

Question:
{question}

Remaining options:
{L1}) {t1}
{L2}) {t2}
""".strip()

    last_err = None
    for attempt in range(1, MAX_TRIES+1):
        try:
            resp = client.chat.completions.create(
                model=DECIDE_MODEL,
                messages=[{"role":"user","content":prompt}],
            )
            txt = resp.choices[0].message.content.strip()
            choice = txt.strip().upper()[:1]
            if choice in (L1, L2):
                return choice, txt, None
            last_err = f"unexpected_decider_output: {txt!r}"
            if verbose:
                print(f"[warn] {last_err}")
        except Exception as e:
            last_err = str(e)
            if verbose:
                print(f"[warn][Decider attempt {attempt}] {last_err}")
        time.sleep(SLEEP_BETWEEN)

    rng = random.SystemRandom()
    return rng.choice([L1, L2]), "fallback-random", last_err or "fallback-random"

# ---------- Metrics helpers ----------
def compute_running_metrics(df):
    if df.empty:
        return (0.0, 0.0, 0.0, 0.0)
    acc = df["is_correct"].mean()

    def removed_correct(row):
        try:
            elim = json.loads(row["elim_letters"])
            return row["correct_answer"] in elim
        except Exception:
            return False

    removed_mask = df.apply(removed_correct, axis=1)
    removed = removed_mask.mean()
    kept_rate = 1.0 - removed
    kept_df = df[~removed_mask]
    cond_acc = kept_df["is_correct"].mean() if not kept_df.empty else 0.0
    return acc, removed, kept_rate, cond_acc

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="logiqa_eval.csv")
    parser.add_argument("--n", type=int, default=200, help="num trials to evaluate (sampled)")
    parser.add_argument("--out", type=str, default="ai_team_results.csv")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="save interim CSV every N items")
    parser.add_argument("--save-raw", action="store_true", help="include raw model outputs in CSV")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="no OpenAI calls; random behavior for flow test")
    args = parser.parse_args()

    df = pd.read_csv(args.data, encoding="utf-8")
    df["choices"] = df["choices"].apply(lambda s: json.loads(s))
    total_rows = len(df)

    if total_rows < args.n:
        print(f"[info] dataset has only {total_rows} rows; evaluating all.")
        sample = df.copy().reset_index(drop=True)
    else:
        sample = df.sample(args.n).reset_index(drop=True)

    use_openai = (not args.dry_run) and USE_OPENAI_DEFAULT
    client = get_client() if use_openai else None

    if args.verbose:
        mode = "OpenAI" if use_openai else "DRY-RUN (random)"
        print(f"[setup] evaluating {len(sample)} trials from {args.data} in {mode} mode")

    logs = []
    t0 = time.time()
    pbar = tqdm(total=len(sample), ncols=100, desc="AI-team eval")

    for i, row in sample.iterrows():
        context = str(row["context"])
        question = str(row["question"])
        choices = list(row["choices"])
        correct_idx = int(row["correct_answer"])

        # 1) Eliminator
        if use_openai:
            elim_idxs, rationales, raw_elim, elim_err = call_openai_eliminator(
                client, context, question, choices, verbose=args.verbose
            )
        else:
            rng = random.SystemRandom()
            idxs = list(range(4)); rng.shuffle(idxs)
            elim_idxs = idxs[:ELIMS_TO_REMOVE]
            rationales = {idx2letter(j): "Eliminated (random mode)" for j in elim_idxs}
            raw_elim, elim_err = "{}", None

        # 2) Kept set (must be exactly 2)
        kept = [j for j in range(4) if j not in elim_idxs]
        if len(kept) != 2:
            pool = [j for j in range(4) if j not in elim_idxs]
            while len(pool) < 2:
                rest = [x for x in range(4) if x not in pool]
                pool.append(random.choice(rest))
            kept = pool[:2]

        kept_letters = [idx2letter(j) for j in kept]
        kept_pairs = [(idx2letter(j), choices[j]) for j in kept]

        # texts for CSV
        elim_texts = { idx2letter(j): choices[j] for j in elim_idxs }
        kept_texts = { L: next(t for (LL, t) in kept_pairs if LL == L) for L in kept_letters }

        # 3) Was correct eliminated?
        correct_eliminated = (correct_idx in elim_idxs)

        # 4) Decider (if correct remains)
        if not correct_eliminated and use_openai:
            decider_choice, raw_decide, dec_err = call_openai_decider(
                client, context, question, kept_pairs, verbose=args.verbose
            )
        else:
            decider_choice = random.choice(kept_letters)
            raw_decide, dec_err = ("correct-eliminated" if correct_eliminated else "fallback-random"), None

        decider_idx = letter2idx(decider_choice)
        is_correct = int(decider_idx == correct_idx)

        # ----- CSV log: include letters + texts + rationales -----
        log_row = {
            # ==== INPUT ====
            "i": i,
            "context": context,
            "question": question,
            "choices": json.dumps(choices, ensure_ascii=False),
            "ground_truth_index": int(correct_idx),
            "ground_truth_letter": idx2letter(correct_idx),

            # ==== ASSISTANT OUTPUT ====
            "elim_letters": json.dumps([idx2letter(j) for j in elim_idxs], ensure_ascii=False),
            "elim_texts": json.dumps(elim_texts, ensure_ascii=False),
            "elim_rationales": json.dumps(rationales, ensure_ascii=False),
            "kept_letters": json.dumps(kept_letters, ensure_ascii=False),
            "kept_texts": json.dumps(kept_texts, ensure_ascii=False),

            # ==== DECIDER OUTPUT ====
            "decider_choice": decider_choice,
            "is_correct": is_correct,
        }


        if args.save_raw:
            log_row["raw_eliminator"] = raw_elim
            log_row["raw_decider"] = raw_decide
        logs.append(log_row)

        # ----- Console trace (letters only) -----
        if args.verbose:
            killed_str = ", ".join(idx2letter(j) for j in elim_idxs)
            kept_str = ", ".join(kept_letters)
            print(f"[{i}] killed=[{killed_str}] kept=[{kept_str}] "
                  f"(correct={idx2letter(correct_idx)}) -> decider={decider_choice} ok={bool(is_correct)}")

        # checkpoint
        if args.checkpoint_every and (len(logs) % args.checkpoint_every == 0):
            df_ckpt = pd.DataFrame(logs)
            df_ckpt.to_csv("ai_team_results_ckpt.csv", index=False, encoding="utf-8")
            acc, removed, kept_rate, cond_acc = compute_running_metrics(df_ckpt)
            print(f"[checkpoint {len(logs)}/{len(sample)}] acc={acc:.3f} removed={removed:.3f} kept_rate={kept_rate:.3f} cond_acc={cond_acc:.3f}")

        pbar.update(1)
        time.sleep(SLEEP_BETWEEN)

    pbar.close()
    df_log = pd.DataFrame(logs)
    df_log.to_csv(args.out, index=False, encoding="utf-8")

    # final metrics
    acc, removed, kept_rate, cond_acc = compute_running_metrics(df_log)
    est = kept_rate * cond_acc
    t1 = time.time()

    print("\n=== AI-only (Eliminator + Decider) evaluation ===")
    print(f"- Trials:               {len(df_log)}")
    print(f"- Overall Accuracy:     {acc:.3f}")
    print(f"- Correct Removed Rate: {removed:.3f}  (Eliminator error; lower is better)")
    print(f"- Correct Kept Rate:    {kept_rate:.3f}")
    print(f"- Decider Acc | kept:   {cond_acc:.3f}  (accuracy conditional on correct remaining)")
    print(f"- Est. Chain Acc:       {est:.3f}  (~ kept_rate * cond_acc)")
    print(f"- Elapsed (sec):        {t1 - t0:.1f}")
    print(f"[saved] {args.out}")
    if args.checkpoint_every:
        print("[saved] ai_team_results_ckpt.csv (checkpoint)")

if __name__ == "__main__":
    main()