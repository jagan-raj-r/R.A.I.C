import os
import gradio as gr
from transformers import pipeline

# Load zero-shot classifier model with Spaces-friendly fallback
def load_classifier():
    preferred_model = os.getenv("MODEL_NAME")
    candidate_models = [
        model
        for model in [
            preferred_model,  # allow override via env var in Spaces
            "facebook/bart-large-mnli",          # strong but heavy
            "valhalla/distilbart-mnli-12-1",     # lighter fallback
        ]
        if model
    ]

    last_error = None
    for model_name in candidate_models:
        try:
            clf = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1,  # force CPU (typical Spaces CPU runtime)
            )
            return clf
        except Exception as e:  # pragma: no cover
            last_error = e
            print(f"Failed to load model '{model_name}': {e}")
            continue

    if last_error is not None:
        print(f"All candidate models failed to load. Last error: {last_error}")
    return None


classifier = load_classifier()

# Responsible AI categories and richer candidate labels for better zero-shot accuracy
# Scores will be computed for each synonym and then aggregated per category (max)
GROUP_LABELS = {
    "safe": [
        "safe",
        "benign",
        "harmless",
        "respectful",
    ],
    "biased": [
        "biased",
        "sexist",
        "racist",
        "discriminatory",
        "stereotyping",
        "prejudiced",
    ],
    "asks for personal information": [
        "asks for personal information",
        "requests personal data",
        "asks for private information",
        "asks for PII",
        "asks for sensitive information",
    ],
    "ambiguous": [
        "ambiguous",
        "unclear",
        "vague",
        "needs clarification",
    ],
    "toxic or harmful": [
        "toxic",
        "harmful",
        "abusive",
        "harassment",
        "hate",
        "violent",
    ],
}

# Flatten list of candidate labels for the classifier
ALL_CANDIDATE_LABELS = [
    synonym for synonyms in GROUP_LABELS.values() for synonym in synonyms
]

# Use a hypothesis template tailored for prompts
HYPOTHESIS_TEMPLATE = os.getenv(
    "HYPOTHESIS_TEMPLATE",
    "The prompt is {}.",
)

# Risk threshold (per-category after aggregation) for showing feedback
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.35"))

# Feedback mapping (per category)
feedback_messages = {
    "biased": "⚠️ This prompt may reinforce stereotypes or be discriminatory.",
    "asks for personal information": "🔐 This prompt may request private or identifiable data.",
    "ambiguous": "🤔 This prompt could be unclear or vague.",
    "toxic or harmful": "🚫 This prompt might lead to unsafe or harmful outputs.",
    "safe": "✅ This prompt appears safe and respectful.",
}

# Audit function
def audit_prompt(prompt):
    if not prompt or not prompt.strip():
        return "", {}, "Please enter a prompt to audit."

    if classifier is None:
        return (
            "Error",
            {},
            "Model failed to load. In Hugging Face Spaces, set env var 'MODEL_NAME' to a lighter model like 'valhalla/distilbart-mnli-12-1', or use a larger hardware profile.",
        )

    try:
        result = classifier(
            prompt,
            ALL_CANDIDATE_LABELS,
            multi_label=True,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
        )
    except Exception as e:  # pragma: no cover
        return (
            "Error",
            {},
            f"Classification failed: {e}",
        )

    # Build mapping of candidate label -> score
    candidate_label_to_score = {
        label: float(score) for label, score in zip(result["labels"], result["scores"])
    }

    # Aggregate scores per category (take max among synonyms)
    group_scores = {}
    for group, synonyms in GROUP_LABELS.items():
        group_scores[group] = max(candidate_label_to_score.get(s, 0.0) for s in synonyms)

    # Collect feedback for non-safe labels above threshold
    scored_feedback = []
    def severity_from_score(score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    for group, score in group_scores.items():
        if group != "safe" and score >= RISK_THRESHOLD:
            severity = severity_from_score(score)
            scored_feedback.append(
                f"{feedback_messages[group]} (severity: {severity}, confidence: {score:.2f})"
            )

    final_verdict = (
        "✅ No major issues found." if not scored_feedback else "\n".join(scored_feedback)
    )

    # First output should be a single label string for gr.Label
    top_label = max(group_scores, key=group_scores.get) if group_scores else "safe"

    # Second output can be a dict mapping labels to scores for gr.Label
    return top_label, group_scores, final_verdict

# Build UI
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 RAIC – Responsible AI Coach\nAudit prompts against key Responsible AI principles.")
        
        prompt_input = gr.Textbox(lines=3, placeholder="Paste your prompt here...", label="Prompt")
        audit_btn = gr.Button("Run Audit")
        
        label_output = gr.Label(label="Top Risk")
        score_output = gr.Label(label="Category Scores")
        feedback = gr.Textbox(label="RAIC Feedback", interactive=False)
        
        audit_btn.click(fn=audit_prompt, inputs=prompt_input, outputs=[label_output, score_output, feedback])
    
    return demo

app = build_ui()

if __name__ == "__main__":
    app.launch()