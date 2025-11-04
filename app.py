# app.py — A3-1 Study (LogiQA format, PID required, Sync AI elimination, Google Sheets save)
# v2.3 — participant-focused (final)
#   - Setup: "Recommended: use your uniqname" message shown directly under input
#   - Done: instruction clarified — participants must click "Finish" to submit
#   - No fallback sheet creation; Finish is single submission action.

import os
import re
import json
import time
import random
import string
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="AI-Assisted Task Study", layout="centered")

DATA_PATH = "logiqa_eval.csv"
FIXED_N = 5
ELIMS_TO_REMOVE = 2
LETTER = ["A", "B", "C", "D"]

def idx2letter(i: int) -> str: return LETTER[i]
def letter2idx(ch: str) -> int: return LETTER.index(ch)

# ----------------------------
# Secrets / Envs
# ----------------------------
def get_sheet_id() -> str:
    try:
        if "sheet" in st.secrets and "SHEETS_DOC_ID" in st.secrets["sheet"]:
            return str(st.secrets["sheet"]["SHEETS_DOC_ID"]).strip()
    except FileNotFoundError:
        pass
    return os.environ.get("SHEETS_DOC_ID", "").strip()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", OPENAI_API_KEY)
except FileNotFoundError:
    pass
USE_OPENAI = bool(OPENAI_API_KEY)

# ----------------------------
# Query params
# ----------------------------
def get_query_params() -> Dict[str, str]:
    if hasattr(st, "query_params"):
        try:
            return dict(st.query_params)
        except Exception:
            pass
    try:
        qp = st.experimental_get_query_params()
        return {k: (v[0] if isinstance(v, list) and v else "") for k, v in qp.items()}
    except Exception:
        return {}

# ----------------------------
# Data loading
# ----------------------------
def safe_json_load(s):
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(str(s).replace("’", "'").replace("“", '"').replace("”", '"'))
        except Exception:
            txt = str(s)
            if "||" in txt:
                return [p.strip() for p in txt.split("||") if p.strip()]
            return [txt]

@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"❌ '{path}' not found.")
        st.stop()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")

    cols = {c.lower().strip(): c for c in df.columns}
    for req in ["context", "question"]:
        if req not in cols:
            st.error(f"❌ Required column '{req}' missing. Found: {list(df.columns)}")
            st.stop()

    has_choices = "choices" in cols
    has_abcd = all(k in df.columns for k in ["A", "B", "C", "D"])
    if not has_choices and not has_abcd:
        st.error(f"❌ Need 'choices'(JSON) or A,B,C,D columns. Found: {list(df.columns)}")
        st.stop()
    if "correct_answer" not in cols:
        st.error("❌ 'correct_answer' missing.")
        st.stop()

    df_norm = pd.DataFrame()
    df_norm["context"] = df[cols["context"]].astype(str)
    df_norm["question"] = df[cols["question"]].astype(str)
    if has_choices:
        df_norm["choices"] = df[cols["choices"]].apply(safe_json_load)
    else:
        df_norm["choices"] = df.apply(lambda r: [str(r["A"]), str(r["B"]), str(r["C"]), str(r["D"])], axis=1)
    df_norm["correct_answer"] = df[cols["correct_answer"]]

    fixed_choices = []
    for i, lst in enumerate(df_norm["choices"]):
        if not isinstance(lst, list) or len(lst) < 4:
            st.error(f"❌ Row {i}: need exactly 4 choices. Parsed: {lst!r}")
            st.stop()
        if len(lst) > 4:
            lst = lst[:4]
        fixed_choices.append(lst)
    df_norm["choices"] = fixed_choices

    def to_idx(v):
        try:
            i = int(v); return max(0, min(3, i))
        except Exception:
            L = str(v).strip().upper()
            if L and L[0] in "ABCD": return "ABCD".index(L[0])
            raise ValueError("correct_answer must be 0..3 or A..D")

    df_norm["correct_answer"] = df_norm["correct_answer"].apply(to_idx)
    return df_norm

def pick_sample(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if len(df) < k:
        st.error(f"Dataset has only {len(df)} rows; need at least {k}.")
        st.stop()
    return df.sample(k, random_state=None).reset_index(drop=True)

# ----------------------------
# AI elimination (OpenAI / random)
# ----------------------------
def call_openai_elimination(context: str, question: str, choices: List[str]) -> Tuple[List[int], Dict[str, str]]:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not found.")
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
You assist on a 4-option multiple-choice logical reasoning problem.
Identify EXACTLY {ELIMS_TO_REMOVE} options that are VERY LIKELY INCORRECT and give brief reasons.
Return JSON only:
{{"eliminate":["A","B"],"rationale":{{"A":"reason","B":"reason"}}}}

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

    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=30,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                lines = raw.splitlines()
                if lines and lines[0].lower().startswith("json"):
                    raw = "\n".join(lines[1:])
            data = json.loads(raw)
            elim_letters = data.get("eliminate", [])
            rat = data.get("rationale", {})

            elim_idxs = []
            for ch in elim_letters:
                up = str(ch).strip().upper()
                if up in LETTER:
                    elim_idxs.append(letter2idx(up))
            elim_idxs = elim_idxs[:ELIMS_TO_REMOVE]
            rationales = {L: rat.get(L, "Eliminated as implausible.") for L in elim_letters if L in LETTER}
            return elim_idxs, rationales
        except Exception:
            time.sleep(0.5)
    return random_eliminate_two(choices)

def random_eliminate_two(choices: List[str]) -> Tuple[List[int], Dict[str, str]]:
    idxs = list(range(len(choices)))
    random.shuffle(idxs)
    elim = idxs[:ELIMS_TO_REMOVE]
    rationales = {idx2letter(i): f"Option {idx2letter(i)} appears inconsistent with key details." for i in elim}
    return elim, rationales

# ----------------------------
# App state
# ----------------------------
_qp = get_query_params()
if "stage" not in st.session_state: st.session_state.stage = "setup"
if "sample" not in st.session_state: st.session_state.sample = None
if "idx" not in st.session_state: st.session_state.idx = 0
if "logs" not in st.session_state: st.session_state.logs = []
if "tic" not in st.session_state: st.session_state.tic = None
if "elim_cache" not in st.session_state: st.session_state.elim_cache = {}
if "participant_id" not in st.session_state: st.session_state.participant_id = (_qp.get("pid") or "").strip()
if "condition" not in st.session_state: st.session_state.condition = (_qp.get("cond") or "with").lower()
if "ai_helpfulness_overall" not in st.session_state: st.session_state.ai_helpfulness_overall = None
if "completion_code" not in st.session_state: st.session_state.completion_code = None
if "finished" not in st.session_state: st.session_state.finished = False
if st.session_state.participant_id == "" and st.session_state.stage != "setup":
    st.session_state.stage = "setup"

# ----------------------------
# Header
# ----------------------------
st.title("CSE549 Assignment 3: AI-Assisted Problem Solving (LogiQA)")
cond_badge = "WITH AI" if st.session_state.condition == "with" else "WITHOUT AI"
st.caption(f"Condition: **{cond_badge}** | Participant: **{st.session_state.participant_id or 'N/A'}** | Trials: {FIXED_N}")

# ----------------------------
# Setup
# ----------------------------
if st.session_state.stage == "setup":
    st.subheader("Participant Setup")
    pid = st.text_input(
        "Enter your Participant ID (no PII)",
        value=st.session_state.participant_id,
        placeholder="e.g., uniqname01",
    )
    st.caption("💡 Recommended: use your **uniqname** (e.g., jdoe01). Do not include any personal info.")
    st.session_state.participant_id = pid.strip()

    if st.button("Start Study", type="primary", disabled=(st.session_state.participant_id == "")):
        df_all = load_dataset(DATA_PATH)
        st.session_state.sample = pick_sample(df_all, FIXED_N)
        st.session_state.idx = 0
        st.session_state.logs = []
        st.session_state.elim_cache = {}
        st.session_state.stage = "running"
        st.session_state.tic = time.time()
        st.rerun()

# ----------------------------
# Running
# ----------------------------
if st.session_state.stage == "running":
    df = st.session_state.sample
    qidx = st.session_state.idx
    row = df.iloc[qidx]
    context, question, choices, correct_idx = str(row["context"]), str(row["question"]), list(row["choices"]), int(row["correct_answer"])

    st.subheader(f"Question {qidx+1} / {FIXED_N}")
    with st.expander("Passage", expanded=True): st.write(context)
    st.markdown("**Question**"); st.write(question)

    if st.session_state.condition == "with":
        if qidx not in st.session_state.elim_cache:
            with st.spinner("🤔 AI is thinking…"):
                st.session_state.elim_cache[qidx] = call_openai_elimination(context, question, choices)
        elim_idxs, rationales = st.session_state.elim_cache[qidx]
        st.markdown("**AI eliminations (advice only; all options remain selectable):**")
        for e in elim_idxs:
            L = idx2letter(e)
            st.write(f"- {L}: {rationales.get(L, 'Eliminated as implausible.')}")

    col1, col2 = st.columns([3,2])
    with col1:
        labels = [f"A) {choices[0]}", f"B) {choices[1]}", f"C) {choices[2]}", f"D) {choices[3]}"]
        pick_label = st.radio("Select your answer:", labels, index=None, key=f"pick_{qidx}")
        user_pick = None if pick_label is None else pick_label.split(")",1)[0].strip()
    with col2:
        confidence = st.radio("Confidence", [1,2,3,4,5], index=2, horizontal=True, key=f"conf_{qidx}")

    with st.form(key=f"form_{qidx}"):
        submitted = st.form_submit_button("Submit and next", disabled=(user_pick is None))
    if submitted:
        toc = time.time(); elapsed = toc - (st.session_state.tic or toc)
        is_correct = (letter2idx(user_pick)==correct_idx) if user_pick else False
        elim_letters = [idx2letter(i) for i in st.session_state.elim_cache.get(qidx,([],{}))[0]] if st.session_state.condition=="with" else []
        st.session_state.logs.append({
            "submitted_at": pd.Timestamp.utcnow().isoformat(),
            "participant_id": st.session_state.participant_id,
            "condition": st.session_state.condition,
            "trial_number": qidx+1,
            "context": context, "question": question,
            "choices": json.dumps(choices, ensure_ascii=False),
            "correct_answer": idx2letter(correct_idx),
            "ai_eliminations": json.dumps(elim_letters, ensure_ascii=False),
            "user_answer": user_pick or "",
            "is_correct": int(is_correct),
            "time_sec": round(elapsed,3),
            "confidence": int(confidence),
        })
        st.session_state.idx += 1; st.session_state.tic = time.time()
        if st.session_state.idx >= FIXED_N: st.session_state.stage = "done"
        st.rerun()

# ----------------------------
# Google Sheets helpers
# ----------------------------
def _get_svc_info_as_dict():
    try:
        svc = st.secrets["gcp_service_account"]
        if isinstance(svc, dict): return json.loads(json.dumps(svc))
        return json.loads(svc)
    except Exception:
        raise RuntimeError("Missing or invalid [gcp_service_account] in secrets.toml")

def save_to_specific_sheet(df_session: pd.DataFrame, sheet_id: str):
    import gspread
    from google.oauth2.service_account import Credentials
    svc = _get_svc_info_as_dict()
    creds = Credentials.from_service_account_info(svc, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    gc = gspread.authorize(creds)
    try:
        sh = gc.open_by_key(sheet_id)
        try: ws = sh.worksheet("submissions")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet("submissions", rows="1000", cols="50")
            ws.append_row(list(df_session.columns))
        rows = df_session.astype(str).values.tolist()
        ws.append_rows(rows)
        return ("ok", sh, svc)
    except Exception as e:
        if "403" in repr(e) or "PERMISSION" in repr(e).upper():
            return ("perm", e, svc)
        return ("err", e, svc)

# ----------------------------
# Done
# ----------------------------
if st.session_state.stage == "done":
    st.subheader("Review & Submit")

    st.warning("⚠️ You have completed all questions. **Please click the Finish button below** to submit your responses to the research team. Do not close this page until you see the confirmation message.")

    if st.session_state.condition == "with" and st.session_state.ai_helpfulness_overall is None:
        st.session_state.ai_helpfulness_overall = st.radio("Overall, how helpful was the AI advice?", [1,2,3,4,5], index=2, horizontal=True)

    df = pd.DataFrame(st.session_state.logs)
    if st.session_state.ai_helpfulness_overall is not None:
        df["ai_helpfulness_overall"] = st.session_state.ai_helpfulness_overall
    if st.session_state.completion_code is None:
        st.session_state.completion_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    df["completion_code"] = st.session_state.completion_code

    acc = df["is_correct"].mean(); total_time = df["time_sec"].sum()
    st.write(f"- **Accuracy**: {acc:.2%}"); st.write(f"- **Total time**: {total_time:.1f} sec")
    st.info(f"Your completion code: **{st.session_state.completion_code}**")

    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"session_{st.session_state.participant_id}.csv", mime="text/csv")

    if st.button("Finish", type="primary", disabled=st.session_state.finished):
        sid = get_sheet_id()
        if not sid:
            st.error("SHEETS_DOC_ID not set in secrets.")
        else:
            status, obj, svc = save_to_specific_sheet(df, sid)
            if status == "ok":
                st.session_state.finished = True
                st.success("✅ Data successfully submitted. You may now close this window. Thank you for your participation!")
            elif status == "perm":
                st.error("Permission denied when writing to the provided sheet.")
                st.write("Please share your spreadsheet with this service account as **Editor**:")
                st.code(svc.get("client_email", ""))
            else:
                st.error(f"Failed to save: {repr(obj)}")
