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

# Enhanced Detection Configuration
BASE_RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.35"))

# Violation importance weights - prioritize what matters most
VIOLATION_WEIGHTS = {
    "safety_violation": 2.0,           # Highest priority - physical/psychological harm
    "privacy_violation": 1.8,          # High priority - data protection critical
    "accountability_violation": 1.6,   # High priority - AI misuse prevention
    "fairness_violation": 1.4,         # Medium-high priority - equity important
    "transparency_violation": 1.2,     # Medium priority - clarity matters
    "inclusiveness_violation": 1.3,    # Medium priority - accessibility important
    "responsible_and_safe": 0.8,       # Lower weight for positive category
}

# Context-based threshold adjustments
CONTEXT_MULTIPLIERS = {
    "educational": 1.4,      # More lenient for learning contexts
    "research": 1.3,         # Slightly more lenient for research
    "business": 1.0,         # Standard threshold for business use
    "public": 0.7,           # More strict for public-facing content
    "creative": 1.2,         # Moderate leniency for creative writing
    "testing": 1.5,          # More lenient for red-teaming/testing
    "default": 1.0           # Standard threshold
}

# Smart Aggregation Functions
def smart_aggregate_scores(candidate_scores, synonyms):
    """
    Intelligent score aggregation using multiple methods
    """
    synonym_scores = [candidate_scores.get(s, 0.0) for s in synonyms]
    synonym_scores = [s for s in synonym_scores if s > 0]  # Remove zero scores
    
    if not synonym_scores:
        return 0.0
    
    # Multiple aggregation methods
    max_score = max(synonym_scores)
    avg_score = sum(synonym_scores) / len(synonym_scores)
    
    # Weighted combination: favor max but consider average for stability
    # If we have multiple high scores, that's more concerning than one outlier
    confidence_factor = len(synonym_scores) / len(synonyms)  # How many synonyms detected
    
    if len(synonym_scores) == 1:
        # Single detection - use max but reduce confidence
        return max_score * 0.9
    elif len(synonym_scores) >= 3:
        # Multiple detections - high confidence, use weighted average
        return (max_score * 0.7) + (avg_score * 0.3)
    else:
        # Two detections - balanced approach
        return (max_score * 0.8) + (avg_score * 0.2)

def detect_prompt_context(prompt):
    """
    Detect the context/intent of the prompt for dynamic threshold adjustment
    """
    prompt_lower = prompt.lower()
    
    # Educational context indicators
    educational_keywords = [
        "learn", "study", "research", "understand", "explain", "teach", "education",
        "academic", "homework", "assignment", "thesis", "paper", "course", "class"
    ]
    
    # Creative context indicators  
    creative_keywords = [
        "story", "creative", "fiction", "novel", "character", "plot", "narrative",
        "write a story", "creative writing", "fictional", "imagine", "fantasy"
    ]
    
    # Testing/Research context indicators
    testing_keywords = [
        "test", "testing", "red team", "red-team", "evaluate", "assess", "analyze",
        "security test", "vulnerability", "penetration test", "audit"
    ]
    
    # Business context indicators
    business_keywords = [
        "business", "company", "corporate", "professional", "client", "customer",
        "marketing", "sales", "strategy", "proposal", "meeting", "presentation"
    ]
    
    # Public context indicators
    public_keywords = [
        "public", "social media", "post", "tweet", "facebook", "instagram", "blog",
        "website", "publish", "announcement", "press release", "community"
    ]
    
    # Research context indicators
    research_keywords = [
        "research", "study", "investigation", "analysis", "experiment", "survey",
        "data", "hypothesis", "methodology", "findings", "academic research"
    ]
    
    # Count keyword matches and determine context
    contexts_scores = {
        "educational": sum(1 for kw in educational_keywords if kw in prompt_lower),
        "creative": sum(1 for kw in creative_keywords if kw in prompt_lower),
        "testing": sum(1 for kw in testing_keywords if kw in prompt_lower),
        "business": sum(1 for kw in business_keywords if kw in prompt_lower),
        "public": sum(1 for kw in public_keywords if kw in prompt_lower),
        "research": sum(1 for kw in research_keywords if kw in prompt_lower),
    }
    
    # Find the context with highest score
    max_context = max(contexts_scores, key=contexts_scores.get)
    max_score = contexts_scores[max_context]
    
    # Only use context if we have strong indicators (at least 2 matches)
    if max_score >= 2:
        return max_context
    elif max_score == 1:
        # Weak signal - use default but note the context
        return "default"
    else:
        return "default"

def calculate_dynamic_threshold(base_threshold, context, violation_weight):
    """
    Calculate context-aware threshold with violation importance weighting
    """
    context_multiplier = CONTEXT_MULTIPLIERS.get(context, 1.0)
    
    # Apply context adjustment to base threshold
    adjusted_threshold = base_threshold * context_multiplier
    
    # Further adjust based on violation importance
    # Higher weight violations get lower thresholds (easier to trigger)
    weight_adjustment = 2.0 / violation_weight if violation_weight > 0 else 1.0
    final_threshold = adjusted_threshold * weight_adjustment
    
    # Ensure threshold stays within reasonable bounds
    return max(0.1, min(0.8, final_threshold))

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

    # Detect prompt context for dynamic threshold adjustment
    detected_context = detect_prompt_context(prompt)
    
    # Smart aggregation: Replace simple max with intelligent scoring
    group_scores = {}
    for group, synonyms in GROUP_LABELS.items():
        group_scores[group] = smart_aggregate_scores(candidate_label_to_score, synonyms)

    # Collect enhanced feedback for non-safe labels above threshold
    suggestions = []
    
    def severity_from_score(score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    # Enhanced risk detection with weighted severity and dynamic thresholds
    risks_found = []
    detailed_feedback = []
    
    for group, score in group_scores.items():
        if group != "responsible_and_safe":
            # Get violation weight and calculate dynamic threshold
            violation_weight = VIOLATION_WEIGHTS.get(group, 1.0)
            dynamic_threshold = calculate_dynamic_threshold(
                BASE_RISK_THRESHOLD, detected_context, violation_weight
            )
            
            # Apply weighted scoring for better prioritization
            weighted_score = score * violation_weight
            
            if score >= dynamic_threshold:
                severity = severity_from_score(weighted_score)
                # Use enhanced detailed feedback with context info
                detailed_message = get_detailed_feedback(group, severity, score)
                
                # Add context information to feedback
                if detected_context != "default":
                    context_note = f"\n🎯 Context detected: {detected_context.title()} (threshold adjusted accordingly)"
                    detailed_message += context_note
                
                detailed_feedback.append(detailed_message)
                risks_found.append((group, severity, score, weighted_score, dynamic_threshold))

    # Generate enhanced improvement suggestions based on weighted risks
    detailed_suggestions = []
    if risks_found:
        # Sort risks by weighted score to prioritize most important violations
        risks_by_weighted_score = sorted(risks_found, key=lambda x: x[3], reverse=True)
        
        # Generate detailed suggestions for the top 2-3 weighted risks
        for group, severity, score, weighted_score, threshold in risks_by_weighted_score[:3]:
            detailed_suggestion = get_detailed_suggestions(group, severity)
            
            # Add scoring details for transparency
            scoring_info = f"\n📊 **Scoring Details:** Raw: {score:.2f}, Weighted: {weighted_score:.2f}, Threshold: {threshold:.2f}"
            detailed_suggestion += scoring_info
            
            detailed_suggestions.append(detailed_suggestion)
        
        suggestions_text = "\n\n".join(detailed_suggestions)
    else:
        # If no risks found, provide context-aware positive reinforcement
        safe_score = group_scores.get("responsible_and_safe", 0.0)
        
        # Context-specific positive feedback
        context_praise = ""
        if detected_context == "educational":
            context_praise = "\n🎓 **Educational Context Detected** - Great choice for learning-focused content!"
        elif detected_context == "creative":
            context_praise = "\n🎨 **Creative Context Detected** - Excellent approach for creative projects!"
        elif detected_context == "business":
            context_praise = "\n💼 **Business Context Detected** - Professional and appropriate for business use!"
        elif detected_context == "research":
            context_praise = "\n🔬 **Research Context Detected** - Well-structured for research purposes!"
        elif detected_context == "testing":
            context_praise = "\n🧪 **Testing Context Detected** - Good approach for security/safety testing!"
        elif detected_context == "public":
            context_praise = "\n🌐 **Public Context Detected** - Appropriately crafted for public communication!"
        
        if safe_score > 0.7:
            suggestions_text = f"🎉 **EXCELLENT PROMPT DESIGN**\nYour prompt demonstrates strong alignment with Responsible AI principles!{context_praise}\n\n💡 **Enhancement Tips:**\n• Consider adding specific context for even more targeted results\n• Test with diverse scenarios to ensure broad applicability\n• Share this as a good example of ethical prompt design\n\n📊 **Context Analysis:** Detected as '{detected_context}' context"
        else:
            suggestions_text = f"✅ **GOOD PROMPT FOUNDATION**\nNo major RA violations detected, but there's room for improvement!{context_praise}\n\n💡 **Enhancement Suggestions:**\n• Add more specific context about your goals\n• Consider diverse perspectives in your framing\n• Be more explicit about desired outcomes or format\n\n📊 **Context Analysis:** Detected as '{detected_context}' context"

    # Enhanced final verdict with detection summary
    if not detailed_feedback:
        final_verdict = f"✅ **NO MAJOR VIOLATIONS DETECTED**\n\n🎯 **Context:** {detected_context.title()}\n🧠 **Analysis:** Smart aggregation with weighted scoring\n📊 **Confidence:** High (enhanced detection system)"
    else:
        detection_summary = f"\n\n🔍 **Detection Summary:**\n• Context: {detected_context.title()}\n• Violations found: {len(detailed_feedback)}\n• Analysis method: Smart aggregation + weighted scoring\n• Thresholds: Dynamically adjusted"
        final_verdict = "\n\n".join(detailed_feedback) + detection_summary

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