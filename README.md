# CSE 594 Assignment #2 — Human-AI Collaboration Task

This repository contains code and data for Assignment #2.  
It includes the following scripts:

- `extract.py` : Extracts and reformats the LogiQA dataset into `logiqa_eval.csv`.
- `evaluate_ai_team.py` (a.k.a. `ai_only.py`) : Runs the AI-only scenario (assistant + decider agents).
- `app.py` : Streamlit interface for the Human-AI collaboration scenario.
- `ai_team_results.csv` : Example output log from an AI-only run (≥200 trials).
- `logiqa_eval.csv` : Reformatted dataset extracted from LogiQA Eval set.
- `wrong picks.xlsx` : Error case summary for analysis.

---

## ⚙️ Environment Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt

2. **Set up your openai api key**