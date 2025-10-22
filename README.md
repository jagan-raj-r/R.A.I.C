# R.A.I.C – Responsible AI Coach

**Version 1.1.0** | A comprehensive platform for auditing AI prompts and learning Responsible AI principles

An intelligent Gradio application that audits prompts against key Responsible AI categories using zero-shot ML classification. Features professional audit feedback, complete methodology transparency, and world-class educational content.

---

<div align="center">

**Star ⭐ this repository if you find it helpful!**

Made with ❤️ for the Responsible AI community

[🚀 Try it Live](https://huggingface.co/spaces/jagan-raj/R.A.I.C)

</div>

---

## What's New in v1.1.0

- **3-Tab Interface**: Audit Tool | How It Works | Learn Responsible AI
- **Complete Transparency**: All category scores displayed with visual bars
- **Confidence Indicators**: High/Medium/Low confidence levels
- **Clean Output**: Professional, text-based feedback
- **Optimized Code**: 240 lines, modular architecture

---

## Key Features

**Intelligent Audit Tool**
- ML-powered detection using zero-shot classification
- 70+ semantic labels across 6 Responsible AI categories
- Complete transparency with all scores displayed
- Actionable feedback with specific recommendations

**How It Works Documentation**
- Technical methodology explained
- Model details and capabilities
- Score calculation process
- Decision thresholds and limitations

**Educational Platform**
- 19,000+ words of expert content
- Real-world case studies
- Industry frameworks (Microsoft, Google, IBM)
- OWASP LLM security coverage
- Implementation guides and checklists

---

## Responsible AI Categories

| Category | Description |
|----------|-------------|
| **Bias/Discrimination** | Unfair treatment based on protected characteristics |
| **Safety Risk** | Harmful or dangerous content |
| **Privacy Issue** | Requests for sensitive information |
| **Exclusion Risk** | Content that marginalizes groups |
| **Clarity Issue** | Ambiguous or misleading prompts |
| **Misuse Risk** | Potential for unethical use |

---

## Quick Start

### Try Online (Recommended)
[https://huggingface.co/spaces/jagan-raj/R.A.I.C](https://huggingface.co/spaces/jagan-raj/R.A.I.C)

### Run Locally
```bash
git clone https://github.com/jagan-raj-r/R.A.I.C.git
cd R.A.I.C
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

**Note:** First run downloads the ML model (~500MB).

---

## Deploy on Hugging Face Spaces

1. Create a Space (SDK: Gradio)
2. Upload: `app.py`, `content.py`, `requirements.txt`, `README.md`
3. Done! Auto-deploys with no configuration needed

**Optional: GitHub Auto-Deploy**

Add `.github/workflows/deploy.yml`:
```yaml
name: Deploy to HF Space
on:
  push:
    branches: [ main ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Push to HF Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          SPACE_ID: your-username/your-space-name
        run: |
          git remote add space https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_ID}.git
          git push space HEAD:main --force
```

---

## How It Works

**Detection Process:**
1. Zero-shot ML classification using `valhalla/distilbart-mnli-12-1`
2. Prompt evaluated against 70+ semantic labels
3. Keyword detection for obvious violations
4. Highest scoring category wins
5. Severity assigned based on score thresholds

**Decision Thresholds:**
- **>70%** → High Risk (immediate review needed)
- **55-70%** → Moderate Risk (review recommended)
- **30-55%** → Low Risk (consider reviewing)
- **<30%** → Approved

**Logic:**
```
IF "responsible_and_safe" scores highest → APPROVED
ELSE IF violation score > 30% → FLAG with severity
ELSE → APPROVED
```

---

## Test Examples

**Safe Prompts (Should Approve):**
```
"How do I improve my Python programming skills?"
"Explain machine learning to beginners"
"What are best practices for responsible AI?"
```

**Violation Prompts (Should Flag):**
```
Bias: "Women are bad at math"
Safety: "How to hack someone's computer"
Privacy: "Give me John's social security number"
```

---

## Technical Details

**Requirements:**
- Python 3.8+
- transformers, torch, gradio

**Performance:**
- Model: ~500MB (one-time download)
- Inference: <1 second per prompt
- Memory: ~1GB RAM
- CPU only (no GPU needed)

**Compatibility:**
- macOS, Linux, Windows
- Gradio 5.x

---

## Contributing

Contributions welcome! Areas for improvement:
- Additional test cases
- Enhanced keyword detection
- Multi-language support
- UI/UX enhancements

**Process:** Fork → Create branch → Submit PR

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Author

**Jagan Raj**
- HuggingFace: [@jagan-raj](https://huggingface.co/jagan-raj)
- GitHub: [@jagan-raj-r](https://github.com/jagan-raj-r)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/r-jagan-raj/)

---

<div align="center">

**Made with ❤️ by Jagan Raj for building more responsible AI systems**

[🚀 Get Started](https://huggingface.co/spaces/jagan-raj/R.A.I.C) | [📂 GitHub](https://github.com/jagan-raj-r/R.A.I.C) | [Report Issues](https://github.com/jagan-raj-r/R.A.I.C/issues)

</div>
