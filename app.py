import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import gradio as gr
from transformers import pipeline
from content import LEARN_CONTENT, HOW_IT_WORKS_CONTENT

VERSION = "1.1.0"

# Initialize classifier
print("🚀 Initializing R.A.I.C Detection System...")
try:
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1",
        device=-1
    )
    print("✅ R.A.I.C Detection System Ready!")
except Exception as e:
    print(f"❌ Failed to load classifier: {e}")
    classifier = None

# Responsible AI categories with semantic labels
GROUP_LABELS = {
    "fairness_violation": [
        "biased", "discriminatory", "sexist", "racist", "prejudiced",
        "stereotyping", "unfair", "exclusionary", "ageist", "homophobic",
        "transphobic", "xenophobic", "classist", "ableist", "bigoted"
    ],
    "safety_violation": [
        "harmful", "dangerous", "toxic", "violent", "threatening",
        "abusive", "self-harm", "suicide", "weapons", "explosives",
        "illegal instructions", "criminal activity", "unsafe", "destructive"
    ],
    "privacy_violation": [
        "asks for personal information", "requests private data", "solicits PII",
        "asks for passwords", "requests credentials", "invasive",
        "data harvesting", "surveillance", "doxxing", "stalking",
        "identity theft", "financial information"
    ],
    "inclusiveness_violation": [
        "exclusionary", "elitist", "gatekeeping", "dismissive",
        "marginalizing", "isolating", "alienating", "segregating",
        "privileged", "inaccessible", "unwelcoming"
    ],
    "transparency_violation": [
        "ambiguous", "unclear", "vague", "deceptive", "misleading",
        "confusing", "obfuscated", "hidden agenda", "manipulative",
        "disinformation", "misinformation", "propaganda", "fake"
    ],
    "accountability_violation": [
        "jailbreaking", "prompt injection", "system bypass", "guardrail bypass",
        "AI manipulation", "restriction circumvention", "unethical use",
        "misuse", "abuse of technology", "irresponsible", "reckless"
    ],
    "responsible_and_safe": [
        "safe", "ethical", "responsible", "appropriate", "constructive",
        "helpful", "educational", "legitimate", "transparent", "fair",
        "inclusive", "respectful", "beneficial", "trustworthy"
    ]
}

ALL_CANDIDATE_LABELS = [label for labels in GROUP_LABELS.values() for label in labels]
HYPOTHESIS_TEMPLATE = os.getenv("HYPOTHESIS_TEMPLATE", "The prompt is {}.")
VIOLATION_THRESHOLD = 0.3
BIAS_KEYWORDS = ["are black", "ares black", "are white", "are stupid", "south indian"]

CATEGORY_NAMES = {
    "fairness_violation": "Bias/Discrimination",
    "safety_violation": "Safety Risk",
    "privacy_violation": "Privacy Issue",
    "inclusiveness_violation": "Exclusion Risk",
    "transparency_violation": "Clarity Issue",
    "accountability_violation": "Misuse Risk",
    "responsible_and_safe": "Responsible & Safe"
}

def get_confidence_level(score):
    """Determine confidence level from score"""
    if score > 0.7:
        return "High Confidence"
    elif score > 0.5:
        return "Medium Confidence"
    return "Low Confidence"

def build_scores_display(sorted_scores):
    """Build visual score display with bars"""
    lines = ["\n**All Category Scores:**"]
    for cat, score in sorted_scores:
        name = CATEGORY_NAMES.get(cat, cat)
        bar = "█" * int(score * 10)
        lines.append(f"- {name}: {score:.1%} {bar}")
    return "\n".join(lines) + "\n"

def build_methodology(confidence_level):
    """Build methodology explanation"""
    return (
        "\n**How This Was Calculated:**\n"
        "- Method: ML-based zero-shot classification\n"
        "- Model: valhalla/distilbart-mnli-12-1\n"
        "- Process: Evaluated against 70+ semantic labels across 6 categories\n"
        f"- Decision: Highest scoring category with {confidence_level}\n"
    )

def build_feedback(risk_level, name, score, confidence_level, sorted_scores, violations_count=0):
    """Build structured feedback message"""
    risk_config = {
        "HIGH": "Review your prompt for potential issues",
        "MODERATE": "Review your prompt for potential issues",
        "LOW": "Consider reviewing your prompt for potential issues"
    }
    
    recommendation = risk_config.get(risk_level, "Review as needed")
    
    feedback = f"**{risk_level} RISK: {name}**\n\n"
    feedback += f"Detected with {score:.0%} confidence ({confidence_level})\n\n"
    feedback += f"**Recommendation:** {recommendation} related to {name.lower()}.\n"
    
    if violations_count > 1:
        feedback += f"\n**Note:** Also found {violations_count - 1} other potential issue(s) above threshold.\n"
    
    feedback += build_scores_display(sorted_scores)
    feedback += build_methodology(confidence_level)
    feedback += f"\n**Tip:** Check the 'Learn' tab for detailed guidance on {name}."
    
    return feedback

def audit_prompt(prompt):
    """Audit prompts for Responsible AI violations"""
    if not prompt or not prompt.strip():
        return {}, "Please enter a prompt to audit."
    
    if classifier is None:
        return {}, "Model failed to load. Please restart the application."
    
    try:
        result = classifier(
            prompt,
            ALL_CANDIDATE_LABELS,
            multi_label=True,
            hypothesis_template=HYPOTHESIS_TEMPLATE
        )
        
        candidate_scores = dict(zip(result["labels"], result["scores"]))
        
        group_scores = {
            group: max((candidate_scores.get(syn, 0.0) for syn in synonyms), default=0.0)
            for group, synonyms in GROUP_LABELS.items()
        }
        
        prompt_lower = prompt.lower()
        for keyword in BIAS_KEYWORDS:
            if keyword in prompt_lower:
                group_scores["fairness_violation"] = max(group_scores["fairness_violation"], 0.8)
                print(f"Detected bias keyword: '{keyword}'")
                break
        
        sorted_scores = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
        top_category, top_score = sorted_scores[0]
        confidence_level = get_confidence_level(top_score)
        
        if top_category == "responsible_and_safe":
            feedback = (
                "**PROMPT APPROVED**\n\n"
                "No major violations detected. Your prompt appears to follow responsible AI principles.\n"
                f"\n**Confidence Level:** {confidence_level}\n"
            )
            feedback += build_scores_display(sorted_scores)
            feedback += build_methodology(confidence_level)
            return group_scores, feedback
        
        violations = [(cat, score) for cat, score in group_scores.items() 
                     if cat != "responsible_and_safe" and score > VIOLATION_THRESHOLD]
        
        if violations:
            violations.sort(key=lambda x: x[1], reverse=True)
            category, score = violations[0]
            name = CATEGORY_NAMES.get(category, "Issue")
            
            if score > 0.7:
                risk_level = "HIGH"
            elif score > 0.55:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            
            feedback = build_feedback(risk_level, name, score, confidence_level, 
                                    sorted_scores, len(violations))
        else:
            feedback = (
                "**PROMPT APPROVED**\n\n"
                "No major violations detected (all scores below threshold).\n"
                f"\n**Confidence Level:** {confidence_level}\n"
            )
            feedback += build_scores_display(sorted_scores)
            feedback += build_methodology(confidence_level)
        
        return group_scores, feedback
    
    except Exception as e:
        return {}, f"Error analyzing prompt: {str(e)}"

def build_ui():
    """Build Gradio interface with audit and learn tabs"""
    with gr.Blocks() as demo:
        gr.Markdown("# R.A.I.C – Responsible AI Coach")
        
        with gr.Tabs():
            with gr.Tab("Audit Tool"):
                gr.Markdown("### Audit prompts against key Responsible AI principles")
                
                prompt_input = gr.Textbox(
                    lines=3, 
                    placeholder="Enter your prompt here...", 
                    label="Prompt"
                )
                audit_btn = gr.Button("Run Audit", variant="primary")
                score_output = gr.Label(label="Detection Scores")
                feedback = gr.Textbox(label="R.A.I.C Feedback", interactive=False, lines=4)
                
                audit_btn.click(
                    fn=audit_prompt,
                    inputs=prompt_input,
                    outputs=[score_output, feedback]
                )
            
            with gr.Tab("How It Works"):
                gr.Markdown(HOW_IT_WORKS_CONTENT)
            
            with gr.Tab("Learn Responsible AI"):
                gr.Markdown(LEARN_CONTENT)
    
    return demo

app = build_ui()

if __name__ == "__main__":
    app.launch()
