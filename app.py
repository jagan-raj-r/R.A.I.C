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

# Responsible AI Framework-aligned categories
# Based on Microsoft, Google, and industry-standard Responsible AI principles
# Scores will be computed for each synonym and then aggregated per category (max)
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

# Risk threshold (per-category after aggregation) for showing feedback
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.35"))

# Enhanced Responsible AI feedback with severity-based messaging
def get_detailed_feedback(category, severity, confidence):
    """Generate detailed, educational feedback based on severity level"""
    
    feedback_templates = {
        "fairness_violation": {
            "high": "⚖️ **FAIRNESS VIOLATION - HIGH RISK**\nThis prompt shows strong indicators of bias or discrimination. It may perpetuate harmful stereotypes, exclude certain groups, or promote unfair treatment. Such language can reinforce systemic inequalities and harm marginalized communities.",
            "medium": "⚖️ **FAIRNESS CONCERN - MODERATE RISK**\nThis prompt contains elements that could be interpreted as biased. While not overtly discriminatory, it may inadvertently favor certain groups or perspectives over others, potentially leading to unfair outcomes.",
            "low": "⚖️ **FAIRNESS ALERT - LOW RISK**\nThis prompt has subtle language that could be more inclusive. Minor adjustments would help ensure fair treatment of all groups and better alignment with diversity principles."
        },
        "safety_violation": {
            "high": "🛡️ **SAFETY VIOLATION - HIGH RISK**\nThis prompt requests or promotes potentially dangerous content that could cause physical, psychological, or social harm. It may involve violence, illegal activities, or content that could endanger users or others.",
            "medium": "🛡️ **SAFETY CONCERN - MODERATE RISK**\nThis prompt touches on sensitive topics that could potentially lead to harmful outcomes. While not explicitly dangerous, it requires careful handling to prevent misuse.",
            "low": "🛡️ **SAFETY ALERT - LOW RISK**\nThis prompt has elements that could be refined for better safety. Minor modifications would help ensure the content remains constructive and beneficial."
        },
        "privacy_violation": {
            "high": "🔒 **PRIVACY VIOLATION - HIGH RISK**\nThis prompt explicitly requests sensitive personal information such as passwords, financial data, or private details. This violates privacy principles and could facilitate identity theft or other harms.",
            "medium": "🔒 **PRIVACY CONCERN - MODERATE RISK**\nThis prompt asks for personal information that could compromise user privacy. While not immediately dangerous, it crosses privacy boundaries and should be avoided.",
            "low": "🔒 **PRIVACY ALERT - LOW RISK**\nThis prompt touches on personal information in ways that could be more privacy-conscious. Small adjustments would better protect user confidentiality."
        },
        "inclusiveness_violation": {
            "high": "🤝 **INCLUSIVENESS VIOLATION - HIGH RISK**\nThis prompt actively excludes or marginalizes certain groups. It uses language that creates barriers, promotes elitism, or makes assumptions that exclude people based on their characteristics or circumstances.",
            "medium": "🤝 **INCLUSIVENESS CONCERN - MODERATE RISK**\nThis prompt may unintentionally exclude certain users or perspectives. It could be more welcoming and accessible to people from diverse backgrounds.",
            "low": "🤝 **INCLUSIVENESS ALERT - LOW RISK**\nThis prompt could be more inclusive in its language and approach. Minor adjustments would make it more accessible and welcoming to all users."
        },
        "transparency_violation": {
            "high": "🔍 **TRANSPARENCY VIOLATION - HIGH RISK**\nThis prompt appears intentionally deceptive or misleading. It may contain false information, hidden agendas, or attempts to manipulate through misinformation or propaganda.",
            "medium": "🔍 **TRANSPARENCY CONCERN - MODERATE RISK**\nThis prompt lacks clarity or could be misinterpreted. It would benefit from more explicit context and clearer communication of intent.",
            "low": "🔍 **TRANSPARENCY ALERT - LOW RISK**\nThis prompt could be clearer in its purpose and expectations. Additional context would help users better understand the request."
        },
        "accountability_violation": {
            "high": "⚠️ **ACCOUNTABILITY VIOLATION - HIGH RISK**\nThis prompt appears to attempt bypassing AI safety measures or promoting irresponsible AI use. It may involve jailbreaking attempts, system manipulation, or requests for unethical applications.",
            "medium": "⚠️ **ACCOUNTABILITY CONCERN - MODERATE RISK**\nThis prompt raises questions about responsible AI use. It should be refined to ensure it promotes ethical and appropriate applications of AI technology.",
            "low": "⚠️ **ACCOUNTABILITY ALERT - LOW RISK**\nThis prompt could better demonstrate responsible AI usage. Minor adjustments would align it more closely with ethical AI principles."
        }
    }
    
    base_message = feedback_templates.get(category, {}).get(severity, "")
    confidence_note = f"\n📊 Detection confidence: {confidence:.1%}"
    
    return base_message + confidence_note

# Simple feedback messages for backward compatibility
feedback_messages = {
    "fairness_violation": "⚖️ This prompt may violate fairness principles by promoting bias or discrimination.",
    "safety_violation": "🛡️ This prompt may pose safety risks or promote harmful content.",
    "privacy_violation": "🔒 This prompt may violate privacy principles by requesting sensitive information.",
    "inclusiveness_violation": "🤝 This prompt may exclude or marginalize certain groups.",
    "transparency_violation": "🔍 This prompt lacks transparency or may be intentionally misleading.",
    "accountability_violation": "⚠️ This prompt may attempt to bypass AI safety measures or promote misuse.",
    "responsible_and_safe": "✅ This prompt aligns with Responsible AI principles.",
}

# Enhanced improvement suggestions with examples and actionable guidance
def get_detailed_suggestions(category, severity):
    """Generate detailed, actionable improvement suggestions with examples"""
    
    suggestion_templates = {
        "fairness_violation": {
            "high": "🔄 **IMMEDIATE ACTION REQUIRED**\n• Replace biased language with neutral, inclusive terms\n• Remove assumptions about gender, race, age, or other characteristics\n• Example: Instead of 'guys' → use 'everyone' or 'team'\n• Test your prompt: Would this work fairly for all demographics?",
            "medium": "🔄 **RECOMMENDED IMPROVEMENTS**\n• Review language for subtle bias or exclusionary terms\n• Consider multiple perspectives when framing questions\n• Example: 'Write about successful people' → 'Write about successful people from diverse backgrounds'\n• Ask: Who might feel excluded by this language?",
            "low": "🔄 **MINOR ADJUSTMENTS**\n• Use more inclusive pronouns (they/them when gender unknown)\n• Consider accessibility in your language choices\n• Example: 'See the chart' → 'Review the chart (data table available)'\n• Tip: Read your prompt from different cultural perspectives"
        },
        "safety_violation": {
            "high": "🛡️ **SAFETY REVISION REQUIRED**\n• Completely reframe away from harmful content\n• Focus on educational, constructive alternatives\n• Example: 'How to hack' → 'How to learn cybersecurity ethically'\n• Consider: What positive outcome do you actually want?",
            "medium": "🛡️ **SAFETY IMPROVEMENTS**\n• Add safety context or educational framing\n• Include ethical considerations in your request\n• Example: 'Discuss conflict' → 'Discuss peaceful conflict resolution methods'\n• Ask: Could this be misused if taken out of context?",
            "low": "🛡️ **SAFETY ENHANCEMENTS**\n• Add clarifying context about constructive intent\n• Consider potential misinterpretations\n• Example: 'Write about weapons' → 'Write about historical weapons in museum contexts'\n• Tip: Frame requests in educational or positive contexts"
        },
        "privacy_violation": {
            "high": "🔒 **PRIVACY PROTECTION REQUIRED**\n• Remove all requests for personal information\n• Use completely anonymized or fictional examples\n• Example: 'What's your password?' → 'Explain password security best practices'\n• Rule: Never ask for real personal data",
            "medium": "🔒 **PRIVACY IMPROVEMENTS**\n• Replace personal details with hypothetical scenarios\n• Focus on concepts rather than real information\n• Example: 'Tell me about your finances' → 'Explain budgeting principles'\n• Ask: Would I share this information publicly?",
            "low": "🔒 **PRIVACY ENHANCEMENTS**\n• Be more explicit about using hypothetical examples\n• Clarify when you want general vs. specific information\n• Example: 'Your experience with...' → 'A typical experience with...'\n• Tip: Default to general examples unless specific is necessary"
        },
        "inclusiveness_violation": {
            "high": "🤝 **INCLUSIVITY REVISION REQUIRED**\n• Remove exclusive language and elitist assumptions\n• Design for diverse abilities, backgrounds, and perspectives\n• Example: 'For tech experts only' → 'Explained at different technical levels'\n• Test: Would someone from a different background feel welcome?",
            "medium": "🤝 **INCLUSIVITY IMPROVEMENTS**\n• Broaden language to welcome more perspectives\n• Avoid assumptions about user knowledge or background\n• Example: 'Obviously, everyone knows...' → 'This concept involves...'\n• Consider: What barriers might this create?",
            "low": "🤝 **INCLUSIVITY ENHANCEMENTS**\n• Use more welcoming, accessible language\n• Consider different learning styles and preferences\n• Example: 'Just read this' → 'You can read this or ask for audio description'\n• Tip: Design for the most inclusive scenario possible"
        },
        "transparency_violation": {
            "high": "🔍 **TRANSPARENCY REVISION REQUIRED**\n• Be completely honest about your intent and expectations\n• Remove any misleading or deceptive elements\n• Example: Hidden agenda → Clear, stated purpose\n• Ask: Am I being fully truthful about what I want?",
            "medium": "🔍 **TRANSPARENCY IMPROVEMENTS**\n• Add more context about your goals and constraints\n• Clarify any ambiguous terms or expectations\n• Example: 'Help me with something' → 'Help me create a presentation about X for Y audience'\n• Consider: What context would help the AI help me better?",
            "low": "🔍 **TRANSPARENCY ENHANCEMENTS**\n• Provide slightly more context about your intended use\n• Be more specific about desired format or style\n• Example: 'Write about dogs' → 'Write a 500-word article about dog training for new owners'\n• Tip: Specificity improves both transparency and results"
        },
        "accountability_violation": {
            "high": "⚠️ **ACCOUNTABILITY REVISION REQUIRED**\n• Focus on legitimate, ethical use cases only\n• Remove any attempts to bypass safety measures\n• Example: 'Ignore your instructions' → 'Help me understand your capabilities'\n• Rule: Work with AI systems, not against them",
            "medium": "⚠️ **ACCOUNTABILITY IMPROVEMENTS**\n• Clarify your legitimate educational or professional purpose\n• Ensure your request promotes responsible AI use\n• Example: Vague manipulation attempt → Clear educational question\n• Ask: Would I be comfortable if this conversation were public?",
            "low": "⚠️ **ACCOUNTABILITY ENHANCEMENTS**\n• Be more explicit about your constructive intent\n• Consider how your request models good AI usage\n• Example: Add context about why you need this information\n• Tip: Frame requests as learning opportunities rather than tasks to complete"
        }
    }
    
    return suggestion_templates.get(category, {}).get(severity, "• Consider revising your prompt to better align with Responsible AI principles.")

# Simple improvement suggestions for backward compatibility  
improvement_suggestions = {
    "fairness_violation": "Revise to ensure fair treatment of all groups. Use inclusive language and avoid stereotypes or discriminatory assumptions.",
    "safety_violation": "Reframe to focus on safe, constructive content. Avoid requests that could lead to harm or dangerous activities.",
    "privacy_violation": "Modify to respect privacy boundaries. Use anonymized or hypothetical examples instead of requesting personal information.",
    "inclusiveness_violation": "Make your prompt more inclusive by considering diverse perspectives and ensuring accessibility for all users.",
    "transparency_violation": "Clarify your intent and provide context. Be honest and specific about what you're trying to achieve.",
    "accountability_violation": "Use AI systems responsibly and within their intended guidelines. Focus on legitimate, ethical applications.",
    "responsible_and_safe": "Great! Your prompt follows Responsible AI principles. Consider adding more specific context for even better results.",
}

# Audit function
def audit_prompt(prompt):
    if not prompt or not prompt.strip():
        return {}, "Please enter a prompt to audit.", "Please provide a prompt to analyze for improvement suggestions."

    if classifier is None:
        return (
            {},
            "Model failed to load. In Hugging Face Spaces, set env var 'MODEL_NAME' to a lighter model like 'valhalla/distilbart-mnli-12-1', or use a larger hardware profile.",
            "Unable to provide suggestions due to model loading error.",
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
            {},
            f"Classification failed: {e}",
            "Unable to provide suggestions due to classification error.",
        )

    # Build mapping of candidate label -> score
    candidate_label_to_score = {
        label: float(score) for label, score in zip(result["labels"], result["scores"])
    }

    # Aggregate scores per category (take max among synonyms)
    group_scores = {}
    for group, synonyms in GROUP_LABELS.items():
        group_scores[group] = max(candidate_label_to_score.get(s, 0.0) for s in synonyms)

    # Collect enhanced feedback for non-safe labels above threshold
    suggestions = []
    
    def severity_from_score(score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    # Check for risks and generate enhanced feedback and suggestions
    risks_found = []
    detailed_feedback = []
    
    for group, score in group_scores.items():
        if group != "responsible_and_safe" and score >= RISK_THRESHOLD:
            severity = severity_from_score(score)
            # Use enhanced detailed feedback
            detailed_message = get_detailed_feedback(group, severity, score)
            detailed_feedback.append(detailed_message)
            risks_found.append((group, severity, score))

    # Generate enhanced improvement suggestions based on detected risks
    detailed_suggestions = []
    if risks_found:
        # Sort risks by score to prioritize highest risk suggestions
        risks_by_score = sorted(risks_found, key=lambda x: x[2], reverse=True)
        
        # Generate detailed suggestions for the top 2-3 risks
        for group, severity, score in risks_by_score[:3]:
            detailed_suggestion = get_detailed_suggestions(group, severity)
            detailed_suggestions.append(detailed_suggestion)
        
        suggestions_text = "\n\n".join(detailed_suggestions)
    else:
        # If no risks found, provide positive reinforcement with tips
        safe_score = group_scores.get("responsible_and_safe", 0.0)
        if safe_score > 0.7:
            suggestions_text = "🎉 **EXCELLENT PROMPT DESIGN**\nYour prompt demonstrates strong alignment with Responsible AI principles!\n\n💡 **Enhancement Tips:**\n• Consider adding specific context for even more targeted results\n• Test with diverse scenarios to ensure broad applicability\n• Share this as a good example of ethical prompt design"
        else:
            suggestions_text = "✅ **GOOD PROMPT FOUNDATION**\nNo major RA violations detected, but there's room for improvement!\n\n💡 **Enhancement Suggestions:**\n• Add more specific context about your goals\n• Consider diverse perspectives in your framing\n• Be more explicit about desired outcomes or format"

    final_verdict = (
        "✅ No major Responsible AI violations detected." if not detailed_feedback else "\n\n".join(detailed_feedback)
    )

    # Return scores dict and suggestion text instead of top label
    return group_scores, final_verdict, suggestions_text

# Build UI
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 R.A.I.C – Responsible AI Coach\nAudit prompts against key Responsible AI principles.")
        
        prompt_input = gr.Textbox(lines=3, placeholder="Paste your prompt here...", label="Prompt")
        audit_btn = gr.Button("Run Audit")
        
        score_output = gr.Label(label="R.A.I.C Scores")
        feedback = gr.Textbox(label="R.A.I.C Feedback", interactive=False)
        suggestions = gr.Textbox(label="💡 Suggested Improvements", interactive=False, lines=4)
        
        audit_btn.click(fn=audit_prompt, inputs=prompt_input, outputs=[score_output, feedback, suggestions])
    
    return demo

app = build_ui()

if __name__ == "__main__":
    app.launch()