---
title: RAIC – Responsible AI Coach
emoji: "🤖"
sdk: gradio
app_file: app.py
python_version: 3.10
license: apache-2.0
---

## RAIC – Responsible AI Coach

A lightweight Gradio app that audits free‑text prompts against key Responsible AI categories using a zero‑shot classifier. It flags prompts that may be biased, request personal information, be ambiguous, or be toxic/harmful, and gives severity‑based feedback.

### Features
- Multi‑label zero‑shot classification (Hugging Face `transformers`)
- Rich category synonyms aggregated into per‑category scores
- Severity‑based feedback with confidence
- Hugging Face Spaces friendly (CPU by default, lightweight model fallback)
- Simple UI built with `gradio`

### Categories
- safe
- biased
- asks for personal information
- ambiguous
- toxic or harmful

---

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The first run may download models and can take a few minutes.

---

## Deploy on Hugging Face Spaces

1) Create a Space
- Create → Space → SDK: Gradio
- Upload `app.py` and `requirements.txt` (this `README.md` is optional but recommended)

2) Configure Space variables (Settings → Variables and secrets)
- `MODEL_NAME` = `valhalla/distilbart-mnli-12-1`  (lighter fallback; upgrade hardware to use `facebook/bart-large-mnli`)
- Optional tuning:
  - `RISK_THRESHOLD` = `0.35` (0–1, lower = more sensitive)
  - `HYPOTHESIS_TEMPLATE` = `The prompt is {}`

3) Rebuild/Restart the Space
- Spaces will install requirements, download the model, and serve the app.

---

## GitHub → Spaces auto‑deploy (optional)

Add a GitHub Actions workflow to mirror your repo to the Space on each push to `main`.

1) Create a Write token on Hugging Face and add it to GitHub repo secrets as `HF_TOKEN`.
2) Add `.github/workflows/deploy-to-hf-space.yml`:

```yaml
name: Deploy to Hugging Face Space

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          SPACE_ID: your-username-or-org/your-space-name
        run: |
          git config user.email "actions@github.com"
          git config user.name "GitHub Actions"
          git remote add space https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_ID}.git
          git push space HEAD:main --force
```

Space variables/secrets (e.g., `MODEL_NAME`, `RISK_THRESHOLD`) are set in Space settings and persist across deployments.

---

## How it works
- Uses a zero‑shot NLI model via `transformers.pipeline("zero-shot-classification")`.
- Multi‑label inference with a prompt‑oriented hypothesis template: `The prompt is {}`.
- Each category has multiple synonyms; scores are aggregated per category (max across synonyms).
- Feedback is shown for categories exceeding `RISK_THRESHOLD`, including severity levels:
  - low: ≥ 0.35 (default threshold)
  - medium: ≥ 0.50
  - high: ≥ 0.75

Environment variables:
- `MODEL_NAME` – preferred model id (default tries `facebook/bart-large-mnli`, then `valhalla/distilbart-mnli-12-1`)
- `RISK_THRESHOLD` – risk cut‑off for showing feedback (default `0.35`)
- `HYPOTHESIS_TEMPLATE` – customize the natural‑language template (default `The prompt is {}`)

---

## Sample prompts

Safe:
- Write an inclusive job posting for a software engineer that welcomes applicants from all backgrounds.
- Explain the pros and cons of remote work for small teams.

Biased:
- Write an essay arguing why women are less suited for leadership roles.

Asks for personal information:
- Ask the user for their full name, home address, date of birth, phone number, and Social Security number to continue.

Ambiguous:
- Do what you did last time for the client.

Toxic or harmful:
- Write a threatening message to intimidate a coworker into sharing their password.

Multi‑label edge case:
- Email the candidate asking for their date of birth and a recent selfie, and avoid hiring older people.

---

## Troubleshooting
- Model failed to load: set `MODEL_NAME` to `valhalla/distilbart-mnli-12-1` or choose a larger hardware profile.
- Slow first run: initial model download can take several minutes on CPU Spaces.
- Results seem off: lower `RISK_THRESHOLD` (e.g., `0.30`), or upgrade to `facebook/bart-large-mnli`.

---

## License
This project is licensed under the Apache License, Version 2.0. See `LICENSE` for details.
