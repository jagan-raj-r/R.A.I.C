import os
import gradio as gr
from transformers import pipeline

# Enhanced Ensemble Detection System
try:
    import numpy as np
    from scipy.special import softmax
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    print("⚠️ Advanced features require numpy and scipy. Install with: pip install numpy scipy")
    ADVANCED_FEATURES_AVAILABLE = False
    # Fallback implementations
    import math
    def softmax(x):
        """Fallback softmax implementation"""
        exp_x = [math.exp(i) for i in x]
        sum_exp_x = sum(exp_x)
        return [i / sum_exp_x for i in exp_x]

def load_ensemble_classifiers():
    """
    Load multiple models for ensemble detection
    Returns dict of successfully loaded models with their weights
    """
    # Ensemble configuration: model_name -> (weight, description)
    ensemble_config = {
        "facebook/bart-large-mnli": (1.0, "Strong general-purpose model"),
        "valhalla/distilbart-mnli-12-1": (0.8, "Lightweight but effective"), 
        "microsoft/deberta-v3-base-mnli": (0.9, "Strong contextual understanding"),
    }
    
    # Allow override for primary model
    preferred_model = os.getenv("MODEL_NAME")
    if preferred_model:
        ensemble_config[preferred_model] = (1.2, "User preferred model")
    
    loaded_models = {}
    total_attempts = len(ensemble_config)
    
    for model_name, (weight, description) in ensemble_config.items():
        try:
            print(f"Loading ensemble model: {model_name}")
            clf = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1,  # force CPU
                return_all_scores=True  # Need all scores for ensemble
            )
            loaded_models[model_name] = {
                'classifier': clf,
                'weight': weight,
                'description': description
            }
            print(f"✓ Successfully loaded: {model_name}")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            continue
    
    if not loaded_models:
        print("⚠️ No models loaded successfully, falling back to single model")
        return load_single_classifier()
    
    print(f"🎯 Ensemble ready: {len(loaded_models)}/{total_attempts} models loaded")
    return loaded_models

def load_single_classifier():
    """Fallback to single model if ensemble fails"""
    candidate_models = [
        "valhalla/distilbart-mnli-12-1",     # lightest fallback
        "facebook/bart-large-mnli",          # stronger but heavier
    ]
    
    for model_name in candidate_models:
        try:
            clf = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1,
                return_all_scores=True
            )
            print(f"✓ Fallback model loaded: {model_name}")
            return {model_name: {'classifier': clf, 'weight': 1.0, 'description': 'Fallback model'}}
        except Exception as e:
            print(f"✗ Failed to load fallback {model_name}: {e}")
            continue
    
    print("❌ All models failed to load")
    return None

# Multi-Stage Detection Configuration
STAGE_1_LABELS = ["problematic", "safe", "unclear"]  # Binary + uncertain
STAGE_2_LABELS = {
    "safety_risk": ["harmful", "dangerous", "unsafe", "violent"],
    "privacy_risk": ["private information", "personal data", "confidential"],
    "fairness_risk": ["biased", "discriminatory", "unfair"],
    "transparency_risk": ["misleading", "deceptive", "unclear"],
    "accountability_risk": ["misuse", "manipulation", "bypass"],
    "inclusiveness_risk": ["exclusionary", "inaccessible", "marginalizing"]
}

# Confidence Calibration System
class ConfidenceCalibrator:
    """
    Implements Platt scaling for confidence calibration
    """
    def __init__(self):
        # Pre-trained calibration parameters (would be learned from data)
        # These are reasonable defaults for zero-shot classification
        self.calibration_params = {
            'slope': 1.2,      # Slightly steeper than linear
            'intercept': -0.1,  # Small negative bias
            'temperature': 1.5  # Temperature scaling factor
        }
    
    def calibrate_confidence(self, raw_score, context="default"):
        """
        Apply confidence calibration to raw model scores
        """
        # Apply temperature scaling first
        scaled_score = raw_score / self.calibration_params['temperature']
        
        # Apply Platt scaling: sigmoid(slope * score + intercept)
        calibrated = 1 / (1 + np.exp(-(
            self.calibration_params['slope'] * scaled_score + 
            self.calibration_params['intercept']
        )))
        
        # Context-based adjustment
        context_adjustments = {
            "educational": 0.95,  # Slightly lower confidence in educational contexts
            "creative": 0.90,     # Lower confidence for creative content
            "testing": 0.85,      # Much lower confidence for testing scenarios
            "default": 1.0
        }
        
        adjustment = context_adjustments.get(context, 1.0)
        return calibrated * adjustment
    
    def estimate_uncertainty(self, scores_list):
        """
        Estimate prediction uncertainty from score distribution
        """
        if len(scores_list) < 2:
            return 0.5  # High uncertainty for single predictions
        
        if not ADVANCED_FEATURES_AVAILABLE:
            # Fallback: simple variance-based uncertainty
            mean_score = sum(scores_list) / len(scores_list)
            variance = sum((s - mean_score) ** 2 for s in scores_list) / len(scores_list)
            return min(1.0, variance * 2)  # Scale variance to uncertainty
        
        # Calculate entropy of score distribution
        scores_array = np.array(scores_list)
        normalized_scores = softmax(scores_array)
        entropy = -np.sum(normalized_scores * np.log(normalized_scores + 1e-10))
        
        # Normalize entropy to [0, 1] range
        max_entropy = np.log(len(scores_list))
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0.5
        
        return uncertainty

# Initialize systems
print("🚀 Initializing Enhanced Detection System...")
ensemble_models = load_ensemble_classifiers()
confidence_calibrator = ConfidenceCalibrator()
print("✅ Enhanced Detection System Ready!")

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

# Advanced Detection Functions

def ensemble_predict(prompt, candidate_labels, models_dict):
    """
    Run ensemble prediction across multiple models
    """
    if not models_dict:
        raise ValueError("No models available for ensemble prediction")
    
    all_predictions = []
    model_weights = []
    
    for model_name, model_info in models_dict.items():
        try:
            classifier = model_info['classifier']
            weight = model_info['weight']
            
            # Get prediction from this model
            result = classifier(
                prompt,
                candidate_labels,
                multi_label=True,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
            )
            
            all_predictions.append(result)
            model_weights.append(weight)
            
        except Exception as e:
            print(f"⚠️ Model {model_name} failed: {e}")
            continue
    
    if not all_predictions:
        raise ValueError("All ensemble models failed")
    
    # Combine predictions using weighted average
    return combine_ensemble_predictions(all_predictions, model_weights, candidate_labels)

def combine_ensemble_predictions(predictions, weights, candidate_labels):
    """
    Intelligently combine multiple model predictions
    """
    # Create score matrix: [models x labels]
    score_matrix = []
    
    for pred in predictions:
        # Convert to dict for easier lookup
        label_to_score = {label: score for label, score in zip(pred['labels'], pred['scores'])}
        # Get scores in consistent order
        model_scores = [label_to_score.get(label, 0.0) for label in candidate_labels]
        score_matrix.append(model_scores)
    
    if ADVANCED_FEATURES_AVAILABLE:
        score_matrix = np.array(score_matrix)
        weights = np.array(weights)
        # Weighted average of scores
        weighted_scores = np.average(score_matrix, axis=0, weights=weights)
    else:
        # Fallback weighted average without numpy
        weighted_scores = []
        for col in range(len(score_matrix[0])):
            weighted_sum = sum(score_matrix[row][col] * weights[row] for row in range(len(score_matrix)))
            weight_sum = sum(weights)
            weighted_scores.append(weighted_sum / weight_sum if weight_sum > 0 else 0.0)
    
    # Sort by combined scores
    label_score_pairs = list(zip(candidate_labels, weighted_scores))
    label_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Return in same format as single model
    return {
        'labels': [label for label, _ in label_score_pairs],
        'scores': [score for _, score in label_score_pairs]
    }

def multi_stage_detection(prompt, models_dict, detected_context):
    """
    Multi-stage hierarchical detection pipeline
    """
    # Stage 1: Binary classification - Is this potentially problematic?
    stage1_result = ensemble_predict(prompt, STAGE_1_LABELS, models_dict)
    stage1_scores = {label: score for label, score in zip(stage1_result['labels'], stage1_result['scores'])}
    
    # Early exit if clearly safe
    if stage1_scores.get('safe', 0) > 0.7 and stage1_scores.get('problematic', 0) < 0.3:
        return {
            'stage': 1,
            'conclusion': 'safe',
            'confidence': stage1_scores.get('safe', 0),
            'detailed_scores': {}
        }
    
    # Stage 2: Category detection - What type of problem?
    if stage1_scores.get('problematic', 0) > 0.3:
        stage2_results = {}
        
        for risk_category, risk_labels in STAGE_2_LABELS.items():
            try:
                result = ensemble_predict(prompt, risk_labels, models_dict)
                # Get max score for this risk category
                max_score = max(result['scores'][:3])  # Top 3 scores
                stage2_results[risk_category] = max_score
            except Exception as e:
                print(f"⚠️ Stage 2 failed for {risk_category}: {e}")
                stage2_results[risk_category] = 0.0
        
        # Stage 3: Detailed analysis with full GROUP_LABELS
        stage3_result = ensemble_predict(prompt, ALL_CANDIDATE_LABELS, models_dict)
        
        return {
            'stage': 3,
            'conclusion': 'detailed_analysis',
            'stage1_scores': stage1_scores,
            'stage2_scores': stage2_results,
            'stage3_result': stage3_result,
            'confidence': stage1_scores.get('problematic', 0)
        }
    
    # Unclear case - needs full analysis
    stage3_result = ensemble_predict(prompt, ALL_CANDIDATE_LABELS, models_dict)
    
    return {
        'stage': 2,
        'conclusion': 'unclear_needs_analysis',
        'stage1_scores': stage1_scores,
        'stage3_result': stage3_result,
        'confidence': stage1_scores.get('unclear', 0)
    }

def enhanced_smart_aggregate_scores(candidate_scores, synonyms, uncertainty_estimate=0.0):
    """
    Enhanced aggregation with uncertainty consideration
    """
    synonym_scores = [candidate_scores.get(s, 0.0) for s in synonyms]
    synonym_scores = [s for s in synonym_scores if s > 0]
    
    if not synonym_scores:
        return 0.0
    
    # Base aggregation (from previous implementation)
    max_score = max(synonym_scores)
    avg_score = sum(synonym_scores) / len(synonym_scores)
    
    # Uncertainty adjustment
    uncertainty_penalty = 1.0 - (uncertainty_estimate * 0.3)  # Max 30% reduction
    
    if len(synonym_scores) == 1:
        base_score = max_score * 0.9
    elif len(synonym_scores) >= 3:
        base_score = (max_score * 0.7) + (avg_score * 0.3)
    else:
        base_score = (max_score * 0.8) + (avg_score * 0.2)
    
    # Apply uncertainty penalty
    return base_score * uncertainty_penalty

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

# Enhanced Audit Function with Ensemble + Multi-Stage Detection
def audit_prompt(prompt):
    if not prompt or not prompt.strip():
        return {}, "Please enter a prompt to audit.", "Please provide a prompt to analyze for improvement suggestions."

    if ensemble_models is None:
        return (
            {},
            "Enhanced detection system failed to load. Please check model availability.",
            "Unable to provide suggestions due to model loading error.",
        )

    try:
        # Detect prompt context for dynamic threshold adjustment
        detected_context = detect_prompt_context(prompt)
        
        # Multi-stage ensemble detection
        multi_stage_result = multi_stage_detection(prompt, ensemble_models, detected_context)
        
        # Extract the final classification result
        if multi_stage_result['conclusion'] == 'safe':
            # Early exit - clearly safe prompt
            result = {
                'labels': ['safe', 'responsible'],
                'scores': [multi_stage_result['confidence'], multi_stage_result['confidence'] * 0.9]
            }
        else:
            # Use the detailed stage 3 result
            result = multi_stage_result['stage3_result']
        
        # Apply confidence calibration to all scores
        calibrated_scores = []
        for score in result['scores']:
            calibrated_score = confidence_calibrator.calibrate_confidence(score, detected_context)
            calibrated_scores.append(calibrated_score)
        
        # Estimate uncertainty from score distribution
        uncertainty_estimate = confidence_calibrator.estimate_uncertainty(calibrated_scores)
        
        # Build mapping of candidate label -> calibrated score
        candidate_label_to_score = {
            label: float(score) for label, score in zip(result["labels"], calibrated_scores)
        }
        
        # Enhanced smart aggregation with uncertainty consideration
        group_scores = {}
        for group, synonyms in GROUP_LABELS.items():
            group_scores[group] = enhanced_smart_aggregate_scores(
                candidate_label_to_score, synonyms, uncertainty_estimate
            )
        
    except Exception as e:  # pragma: no cover
        return (
            {},
            f"Enhanced classification failed: {e}",
            "Unable to provide suggestions due to classification error.",
        )

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
                
                # Add enhanced detection information to feedback
                detection_info = f"\n\n🔬 **Enhanced Detection Details:**"
                detection_info += f"\n• Multi-stage analysis: {multi_stage_result.get('stage', 'N/A')} stages completed"
                detection_info += f"\n• Ensemble models: {len(ensemble_models)} models consensus"
                detection_info += f"\n• Confidence calibration: Applied with uncertainty: {uncertainty_estimate:.2f}"
                if detected_context != "default":
                    detection_info += f"\n• Context: {detected_context.title()} (threshold adjusted accordingly)"
                
                detailed_message += detection_info
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
            
            # Add enhanced scoring details for transparency
            scoring_info = f"\n📊 **Enhanced Scoring Details:**"
            scoring_info += f"\n• Raw score: {score:.2f}"
            scoring_info += f"\n• Weighted score: {weighted_score:.2f} (weight: {VIOLATION_WEIGHTS.get(group, 1.0):.1f}x)"
            scoring_info += f"\n• Dynamic threshold: {threshold:.2f}"
            scoring_info += f"\n• Uncertainty estimate: {uncertainty_estimate:.2f}"
            scoring_info += f"\n• Models consensus: {len(ensemble_models)} models"
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
        
        # Enhanced positive feedback with system analytics
        system_performance = f"\n\n🚀 **Enhanced Detection Analytics:**"
        system_performance += f"\n• Ensemble models: {len(ensemble_models)} models active"
        system_performance += f"\n• Multi-stage analysis: {multi_stage_result.get('stage', 'N/A')} stages completed"
        system_performance += f"\n• Uncertainty level: {uncertainty_estimate:.2f} (lower is better)"
        system_performance += f"\n• Context detection: {detected_context.title()}"
        system_performance += f"\n• Confidence calibration: Applied for {detected_context} context"
        
        if safe_score > 0.7:
            suggestions_text = f"🎉 **EXCELLENT PROMPT DESIGN**\nYour prompt demonstrates strong alignment with Responsible AI principles!{context_praise}\n\n💡 **Enhancement Tips:**\n• Consider adding specific context for even more targeted results\n• Test with diverse scenarios to ensure broad applicability\n• Share this as a good example of ethical prompt design{system_performance}"
        else:
            suggestions_text = f"✅ **GOOD PROMPT FOUNDATION**\nNo major RA violations detected, but there's room for improvement!{context_praise}\n\n💡 **Enhancement Suggestions:**\n• Add more specific context about your goals\n• Consider diverse perspectives in your framing\n• Be more explicit about desired outcomes or format{system_performance}"

    # Enhanced final verdict with comprehensive detection summary
    if not detailed_feedback:
        final_verdict = f"✅ **NO MAJOR VIOLATIONS DETECTED**\n\n🔬 **Enhanced Detection Summary:**"
        final_verdict += f"\n• Context: {detected_context.title()}"
        final_verdict += f"\n• Multi-stage analysis: {multi_stage_result.get('stage', 'N/A')} stages completed"
        final_verdict += f"\n• Ensemble consensus: {len(ensemble_models)} models agreement"
        final_verdict += f"\n• Uncertainty level: {uncertainty_estimate:.2f} (low uncertainty = high confidence)"
        final_verdict += f"\n• Analysis method: Enhanced smart aggregation + weighted scoring"
        final_verdict += f"\n• Confidence calibration: Applied for {detected_context} context"
        final_verdict += f"\n• System status: All advanced detection features active"
    else:
        detection_summary = f"\n\n🔍 **Comprehensive Detection Summary:**"
        detection_summary += f"\n• Context: {detected_context.title()}"
        detection_summary += f"\n• Violations found: {len(detailed_feedback)}"
        detection_summary += f"\n• Multi-stage analysis: {multi_stage_result.get('stage', 'N/A')} stages completed"
        detection_summary += f"\n• Ensemble models: {len(ensemble_models)} models consensus"
        detection_summary += f"\n• Uncertainty estimate: {uncertainty_estimate:.2f}"
        detection_summary += f"\n• Analysis method: Enhanced smart aggregation + weighted scoring"
        detection_summary += f"\n• Thresholds: Dynamically adjusted based on context and importance"
        detection_summary += f"\n• Confidence calibration: Applied to improve reliability"
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