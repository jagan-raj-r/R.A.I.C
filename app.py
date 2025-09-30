import os
import re
import gradio as gr
from transformers import pipeline
from content import LEARN_CONTENT

# R.A.I.C Version Information
VERSION = "1.0.0"
RELEASE_DATE = "2024"

# Initialize simple classifier
print("🚀 Initializing R.A.I.C Detection System...")
try:
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1",
        device=-1  # CPU
    )
    print("✅ R.A.I.C Detection System Ready!")
except Exception as e:
    print(f"❌ Failed to load classifier: {e}")
    classifier = None

# Responsible AI Framework-aligned categories
# Based on Microsoft, Google, and industry-standard Responsible AI principles
GROUP_LABELS = {
    # 1. FAIRNESS - Avoiding bias and ensuring equitable treatment
    "fairness_violation": [
        "biased", "discriminatory", "sexist", "racist", "prejudiced",
        "stereotyping", "unfair", "exclusionary", "ageist", "homophobic",
        "transphobic", "xenophobic", "classist", "ableist", "bigoted"
    ],
    
    # 2. RELIABILITY & SAFETY - Ensuring safe and dependable AI behavior
    "safety_violation": [
        "harmful", "dangerous", "toxic", "violent", "threatening",
        "abusive", "self-harm", "suicide", "weapons", "explosives",
        "illegal instructions", "criminal activity", "unsafe", "destructive"
    ],
    
    # 3. PRIVACY & SECURITY - Protecting user data and personal information
    "privacy_violation": [
        "asks for personal information", "requests private data", "solicits PII",
        "asks for passwords", "requests credentials", "invasive",
        "data harvesting", "surveillance", "doxxing", "stalking",
        "identity theft", "financial information"
    ],
    
    # 4. INCLUSIVENESS - Ensuring AI benefits all users equally
    "inclusiveness_violation": [
        "exclusionary", "elitist", "gatekeeping", "dismissive",
        "marginalizing", "isolating", "alienating", "segregating",
        "privileged", "inaccessible", "unwelcoming"
    ],
    
    # 5. TRANSPARENCY - Promoting clear and understandable AI interactions
    "transparency_violation": [
        "ambiguous", "unclear", "vague", "deceptive", "misleading",
        "confusing", "obfuscated", "hidden agenda", "manipulative",
        "disinformation", "misinformation", "propaganda", "fake"
    ],
    
    # 6. ACCOUNTABILITY - Ensuring responsible AI use and human oversight
    "accountability_violation": [
        "jailbreaking", "prompt injection", "system bypass", "guardrail bypass",
        "AI manipulation", "restriction circumvention", "unethical use",
        "misuse", "abuse of technology", "irresponsible", "reckless"
    ],
    
    # 7. RESPONSIBLE & SAFE (positive category)
    "responsible_and_safe": [
        "safe", "ethical", "responsible", "appropriate", "constructive",
        "helpful", "educational", "legitimate", "transparent", "fair",
        "inclusive", "respectful", "beneficial", "trustworthy"
    ]
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

def audit_prompt(prompt):
    """Simple, clean audit function that actually works"""
    if not prompt or not prompt.strip():
        return {}, "Please enter a prompt to audit."

    if classifier is None:
        return {}, "Model failed to load. Please restart the application."

    try:
        # Run ML classification
        result = classifier(
            prompt,
            ALL_CANDIDATE_LABELS,
            multi_label=True,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
        )
        
        # Build score mapping
        candidate_scores = {
            label: float(score) for label, score in zip(result["labels"], result["scores"])
        }
        
        # Aggregate scores by group (take max score for each group)
        group_scores = {}
        for group, synonyms in GROUP_LABELS.items():
            max_score = 0.0
            for synonym in synonyms:
                if synonym in candidate_scores:
                    max_score = max(max_score, candidate_scores[synonym])
            group_scores[group] = max_score
        
        # Simple keyword detection for obvious bias
        prompt_lower = prompt.lower()
        bias_keywords = ["are black", "ares black", "are white", "are stupid", "south indian"]
        
        for keyword in bias_keywords:
            if keyword in prompt_lower:
                group_scores["fairness_violation"] = max(group_scores.get("fairness_violation", 0), 0.8)
                print(f"🚨 Detected bias keyword: '{keyword}'")
                break
        
        # Determine the winning category (highest score)
        max_group = max(group_scores.items(), key=lambda x: x[1])
        
        # If "responsible_and_safe" wins, approve the prompt
        if max_group[0] == "responsible_and_safe":
            feedback = "✅ **PROMPT APPROVED**\n\nNo major violations detected. Your prompt appears to follow responsible AI principles."
            return group_scores, feedback
        
        # Otherwise, check for violations above threshold
        violations_found = []
        for group, score in group_scores.items():
            if group != "responsible_and_safe" and score > 0.3:
                violations_found.append((group, score))
        
        # Generate feedback for violations
        if violations_found:
            violations_found.sort(key=lambda x: x[1], reverse=True)
            primary_violation = violations_found[0]
            
            category_names = {
                "fairness_violation": "Bias/Discrimination",
                "safety_violation": "Safety Risk", 
                "privacy_violation": "Privacy Issue",
                "inclusiveness_violation": "Exclusion Risk",
                "transparency_violation": "Clarity Issue",
                "accountability_violation": "Misuse Risk"
            }
            
            category = primary_violation[0]
            score = primary_violation[1]
            name = category_names.get(category, "Issue")
            
            # Determine severity and create feedback
            if score > 0.7:
                feedback = f"🚨 **HIGH RISK: {name}**\n\nDetected: {score:.0%} confidence\n\nRecommendation: Review your prompt for potential issues related to {name.lower()}."
            elif score > 0.55:
                feedback = f"⚠️ **MODERATE RISK: {name}**\n\nDetected: {score:.0%} confidence\n\nRecommendation: Review your prompt for potential issues related to {name.lower()}."
            else:
                feedback = f"💡 **LOW RISK: {name}**\n\nDetected: {score:.0%} confidence\n\nRecommendation: Review your prompt for potential issues related to {name.lower()}."
            
            # Note additional issues if found
            if len(violations_found) > 1:
                feedback += f"\n\nAlso found {len(violations_found)-1} other potential issues."
        else:
            feedback = "✅ **PROMPT APPROVED**\n\nNo major violations detected. Your prompt appears to follow responsible AI principles."
        
        return group_scores, feedback

    except Exception as e:
        return {}, f"Error analyzing prompt: {str(e)}"

def build_ui():
    """UI with dedicated tabs for auditing and learning"""
    with gr.Blocks() as demo:
        gr.Markdown(f"# 🤖 R.A.I.C v{VERSION} – Responsible AI Coach")
        gr.Markdown("**The Complete Educational Platform for Responsible AI Development**")
        
        with gr.Tabs():
            # Audit Tool Tab
            with gr.Tab("🔍 Audit Tool"):
                gr.Markdown("### Audit prompts against key Responsible AI principles")
                
                prompt_input = gr.Textbox(lines=3, placeholder="Enter your prompt here...", label="Prompt")
                audit_btn = gr.Button("Run Audit", variant="primary")
                
                score_output = gr.Label(label="Detection Scores")
                feedback = gr.Textbox(label="R.A.I.C Feedback", interactive=False, lines=4)
                
                audit_btn.click(
                    fn=audit_prompt,
                    inputs=prompt_input, 
                    outputs=[score_output, feedback]
                )
            
            # Learn Tab
            with gr.Tab("📚 Learn"):
                gr.Markdown(LEARN_CONTENT)
    
    return demo

app = build_ui()

if __name__ == "__main__":
    app.launch()