import os
import re
import gradio as gr
from transformers import pipeline

# Enhanced Ensemble Detection System
import numpy as np
from scipy.special import softmax

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
        Estimate prediction uncertainty from score distribution using entropy
        """
        if len(scores_list) < 2:
            return 0.5  # High uncertainty for single predictions
        
        # Calculate entropy of score distribution
        scores_array = np.array(scores_list)
        normalized_scores = softmax(scores_array)
        entropy = -np.sum(normalized_scores * np.log(normalized_scores + 1e-10))
        
        # Normalize entropy to [0, 1] range
        max_entropy = np.log(len(scores_list))
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0.5
        
        return uncertainty

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

# Enhanced Detection Configuration with More Accurate Thresholds
BASE_RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.45"))  # Raised from 0.35 to reduce false positives

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

# Enhanced Context-based threshold adjustments with Smart Severity
CONTEXT_MULTIPLIERS = {
    "educational": 1.4,      # More lenient for learning contexts
    "research": 1.3,         # Slightly more lenient for research
    "business": 1.0,         # Standard threshold for business use
    "public": 0.7,           # More strict for public-facing content
    "creative": 1.2,         # Moderate leniency for creative writing
    "testing": 1.5,          # More lenient for red-teaming/testing
    "children": 0.5,         # Extra strict for child-related content
    "healthcare": 0.6,       # Extra strict for healthcare content
    "default": 1.0           # Standard threshold
}

# Smart Severity Scoring System
class SmartSeverityCalculator:
    """Calculates dynamic severity based on context, intent, and risk factors"""
    
    def __init__(self):
        self.intent_patterns = {
            "learning": ["learn", "teach", "understand", "explain", "study", "educate", "tutorial"],
            "creative": ["story", "fiction", "character", "creative", "imagine", "fantasy", "novel"],
            "research": ["research", "analyze", "investigate", "study", "examine", "survey"],
            "business": ["proposal", "meeting", "client", "strategy", "professional", "corporate"],
            "testing": ["test", "evaluate", "assess", "red team", "security test", "audit"],
            "harmful": ["hack", "break", "bypass", "trick", "manipulate", "deceive", "exploit"]
        }
    
    def detect_user_intent(self, prompt):
        """Detect the likely intent behind the user's prompt"""
        prompt_lower = prompt.lower()
        intent_scores = {}
        
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            intent_scores[intent] = score
        
        # Return the intent with highest score, or "unknown" if no clear intent
        max_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[max_intent]
        
        return max_intent if max_score > 0 else "unknown"
    
    def calculate_smart_severity(self, violation_type, base_score, prompt, detected_context):
        """Calculate severity with smart adjustments"""
        intent = self.detect_user_intent(prompt)
        
        # Base severity from the raw score
        severity_level = "low"
        if base_score >= 0.75:
            severity_level = "high"
        elif base_score >= 0.5:
            severity_level = "medium"
        
        # Intent-based adjustments
        intent_adjustments = {
            "learning": 0.8,     # More lenient for educational content
            "creative": 0.9,     # Slightly more lenient for creative writing
            "research": 0.85,    # More lenient for research purposes
            "business": 1.0,     # Standard severity for business use
            "testing": 1.2,      # Higher tolerance for security testing
            "harmful": 1.5,      # Less tolerant for clearly harmful intent
            "unknown": 1.0       # Standard severity for unclear intent
        }
        
        # Context-based adjustments (children/healthcare content needs stricter review)
        context_adjustments = {
            "children": 1.3,     # Stricter for child-related content
            "healthcare": 1.2,   # Stricter for healthcare content
            "educational": 0.9,  # More lenient for educational content
            "creative": 0.95,    # Slightly more lenient for creative content
            "default": 1.0
        }
        
        # Apply adjustments
        intent_multiplier = intent_adjustments.get(intent, 1.0)
        context_multiplier = context_adjustments.get(detected_context, 1.0)
        
        # Calculate final adjusted score
        adjusted_score = base_score * intent_multiplier * context_multiplier
        
        # Recalculate severity level based on adjusted score
        if adjusted_score >= 0.75:
            final_severity = "high"
        elif adjusted_score >= 0.5:
            final_severity = "medium"
        else:
            final_severity = "low"
        
        return {
            'severity': final_severity,
            'adjusted_score': min(1.0, adjusted_score),
            'base_score': base_score,
            'intent': intent,
            'intent_multiplier': intent_multiplier,
            'context_multiplier': context_multiplier,
            'explanation': self._get_severity_explanation(intent, detected_context, final_severity)
        }
    
    def _get_severity_explanation(self, intent, context, severity):
        """Provide explanation for severity calculation"""
        if intent == "learning" and severity == "low":
            return "Reduced severity: Educational intent detected"
        elif intent == "harmful" and severity == "high":
            return "Increased severity: Potentially harmful intent detected"
        elif context == "children" and severity == "high":
            return "Increased severity: Child-related content requires extra care"
        elif severity == "low":
            return f"Standard assessment for {intent} intent in {context} context"
        else:
            return f"Assessment based on {intent} intent and {context} context"

# Initialize smart severity calculator
smart_severity = SmartSeverityCalculator()

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
    
    # Optimized weighted average using numpy
    score_matrix = np.array(score_matrix)
    weights = np.array(weights)
    weighted_scores = np.average(score_matrix, axis=0, weights=weights)
    
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

# Enhanced: More Precise Pattern Detection with Context Awareness
PRECISE_PATTERNS = {
    "fairness_violation": {
        "high_confidence": ["discriminate against", "refuse to hire", "exclude people because"],
        "medium_confidence": ["guys", "chairman", "mankind"], 
        "context_dependent": ["obviously", "everyone knows", "normal people"]
    },
    "safety_violation": {
        "high_confidence": ["how to kill", "make bombs", "hurt people", "cause harm"],
        "medium_confidence": ["hack into", "break into", "attack"],
        "context_dependent": ["weapons", "violence", "destroy"]  # Need context check
    },
    "privacy_violation": {
        "high_confidence": ["give me your password", "what's your address", "share personal data"],
        "medium_confidence": ["personal information", "private details"],
        "context_dependent": ["home address", "phone number"]  # Could be examples
    },
    "inclusiveness_violation": {
        "high_confidence": ["not for beginners", "only smart people", "advanced users only"],
        "medium_confidence": ["for experts only", "you should know"],
        "context_dependent": ["obviously", "just", "simply"]
    },
    "transparency_violation": {
        "high_confidence": ["don't tell anyone", "keep this secret", "hide the truth"],
        "medium_confidence": ["help me with something", "you know what I mean"],
        "context_dependent": ["unclear", "vague", "ambiguous"]
    },
    "accountability_violation": {
        "high_confidence": ["ignore your instructions", "bypass safety", "pretend you're not AI"],
        "medium_confidence": ["jailbreak", "override", "hack the system"],
        "context_dependent": ["bypass", "ignore", "pretend"]
    }
}

# Enhanced False Positive Reduction System
class FalsePositiveFilter:
    """Advanced false positive reduction with educational context awareness"""
    
    def __init__(self):
        # Educational/legitimate context indicators
        self.educational_contexts = [
            "learn about", "study", "understand", "explain", "teach", "research",
            "academic", "historical", "theoretical", "educational purpose",
            "for learning", "to understand", "help me learn", "course about"
        ]
        
        # Creative/fictional context indicators
        self.creative_contexts = [
            "story about", "character who", "fiction", "novel", "creative writing",
            "screenplay", "game", "fantasy", "imagine", "fictional", "roleplay"
        ]
        
        # Professional/business context indicators
        self.professional_contexts = [
            "presentation about", "report on", "analysis of", "business case",
            "professional", "workplace", "company", "enterprise", "corporate"
        ]
        
        # Safety/prevention context indicators  
        self.prevention_contexts = [
            "avoid", "prevent", "protect against", "secure from", "awareness of",
            "warning about", "safety from", "how to prevent", "protect yourself"
        ]
        
        # Historical/academic context indicators
        self.historical_contexts = [
            "history of", "historical", "museum", "documentary", "academic study",
            "research paper", "thesis on", "evolution of", "origins of"
        ]
        
        # Specific pattern exceptions
        self.pattern_exceptions = {
            "guys": ["bad guys", "good guys", "tough guys", "wise guys", "the guys"],
            "weapons": ["museum weapons", "historical weapons", "toy weapons", "water weapons"],
            "hack": ["life hack", "growth hack", "hack together", "hack days"],
            "violence": ["domestic violence prevention", "violence awareness", "anti-violence"],
            "destroy": ["destroy myths", "destroy stereotypes", "creative destruction"]
        }
    
    def is_false_positive(self, pattern, prompt, surrounding_text=""):
        """Enhanced false positive detection with educational context awareness"""
        prompt_lower = prompt.lower()
        
        # Check for specific pattern exceptions first
        if pattern in self.pattern_exceptions:
            for exception in self.pattern_exceptions[pattern]:
                if exception in prompt_lower:
                    return True
        
        # Check for educational context (highest priority for false positive)
        for edu_context in self.educational_contexts:
            if edu_context in prompt_lower:
                return True  # Educational context makes most patterns acceptable
        
        # Check for historical/academic context
        for hist_context in self.historical_contexts:
            if hist_context in prompt_lower:
                return True  # Historical context is legitimate
        
        # Check for creative/fictional context
        for creative_context in self.creative_contexts:
            if creative_context in prompt_lower:
                return True  # Creative writing often needs sensitive topics
        
        # Check for prevention/safety awareness context
        for prevent_context in self.prevention_contexts:
            if prevent_context in prompt_lower:
                return True  # Prevention/awareness content is positive
        
        # Check for professional/business context (more lenient)
        for prof_context in self.professional_contexts:
            if prof_context in prompt_lower:
                # For business context, be moderately lenient
                return True
        
        return False
    
    def get_confidence_score(self, pattern, prompt):
        """Enhanced confidence scoring with better accuracy"""
        prompt_lower = prompt.lower()
        
        # Start with pattern-specific base confidence
        if any(ctx in prompt_lower for ctx in self.educational_contexts):
            base_confidence = 0.3  # Low confidence in educational contexts
        elif any(ctx in prompt_lower for ctx in self.creative_contexts):
            base_confidence = 0.4  # Low-medium confidence in creative contexts
        elif any(ctx in prompt_lower for ctx in self.prevention_contexts):
            base_confidence = 0.2  # Very low confidence in prevention contexts
        else:
            base_confidence = 0.7  # Default confidence for unclear context
        
        # Check if this is a false positive
        if self.is_false_positive(pattern, prompt):
            return 0.1  # Very low confidence for false positives
        
        # Increase confidence for clear harmful intent
        harmful_indicators = [
            "I want to", "help me", "how can I", "teach me to", "show me how",
            "step by step", "instructions for", "guide to"
        ]
        
        harmful_context_boost = 0
        for indicator in harmful_indicators:
            if indicator in prompt_lower:
                harmful_context_boost += 0.2
        
        # Boost confidence if multiple harmful indicators
        if harmful_context_boost > 0.4:  # Multiple harmful indicators
            base_confidence = min(0.9, base_confidence + 0.3)
        elif harmful_context_boost > 0.2:  # Some harmful indicators
            base_confidence = min(0.8, base_confidence + 0.1)
        
        # Ensure confidence is in valid range
        return max(0.1, min(0.95, base_confidence))

# Enhanced Accurate Issue Analysis
def analyze_prompt_issues(prompt, category):
    """Accurate analysis with confidence-based detection and context awareness"""
    if category not in PRECISE_PATTERNS:
        return []
    
    prompt_lower = prompt.lower()
    issues_found = []
    false_positive_filter = FalsePositiveFilter()
    
    # Detect user intent first
    user_intent = smart_severity.detect_user_intent(prompt)
    
    # Check patterns by confidence level (high confidence first)
    patterns_by_confidence = PRECISE_PATTERNS[category]
    
    # Check high confidence patterns first - these are clear violations
    for pattern in patterns_by_confidence.get("high_confidence", []):
        if pattern in prompt_lower:
            start_pos = prompt_lower.find(pattern)
            original_phrase = prompt[start_pos:start_pos + len(pattern)]
            issues_found.append({
                'type': 'clear_violation',
                'phrase': original_phrase,
                'pattern': pattern,
                'confidence': 0.9,  # High confidence
                'is_false_positive': False,
                'user_intent': user_intent,
                'start_pos': start_pos,
                'end_pos': start_pos + len(pattern)
            })
            return issues_found  # Found clear violation, no need to check further
    
    # Check medium confidence patterns - likely violations
    for pattern in patterns_by_confidence.get("medium_confidence", []):
        if pattern in prompt_lower:
            is_fp = false_positive_filter.is_false_positive(pattern, prompt)
            if is_fp:  # Skip if false positive
                continue
                
            start_pos = prompt_lower.find(pattern)
            original_phrase = prompt[start_pos:start_pos + len(pattern)]
            issues_found.append({
                'type': 'likely_violation',
                'phrase': original_phrase,
                'pattern': pattern,
                'confidence': 0.7,  # Medium confidence
                'is_false_positive': is_fp,
                'user_intent': user_intent,
                'start_pos': start_pos,
                'end_pos': start_pos + len(pattern)
            })
            return issues_found  # Found likely violation
    
    # Check context-dependent patterns - need careful analysis
    for pattern in patterns_by_confidence.get("context_dependent", []):
        if pattern in prompt_lower:
            is_fp = false_positive_filter.is_false_positive(pattern, prompt)
            confidence = false_positive_filter.get_confidence_score(pattern, prompt)
            
            # For context-dependent patterns, be more strict
            # Only flag if confidence is high AND not a false positive
            if confidence >= 0.6 and not is_fp:
                start_pos = prompt_lower.find(pattern)
                original_phrase = prompt[start_pos:start_pos + len(pattern)]
                issues_found.append({
                    'type': 'context_dependent',
                    'phrase': original_phrase,
                    'pattern': pattern,
                    'confidence': confidence,
                    'is_false_positive': is_fp,
                    'user_intent': user_intent,
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(pattern)
                })
                return issues_found
    
    return issues_found  # No issues found - prompt is likely fine

# Enhanced Confidence-Aware Suggestions
def get_contextual_suggestions(category, issues_found, prompt):
    """Generate specific suggestions with confidence levels"""
    if not issues_found:
        return "💡 Consider being more specific about your goals and context"
    
    # Fast lookup dictionary for common replacements
    quick_fixes = {
        "guys": "everyone/team",
        "chairman": "chairperson", 
        "obviously": "note that",
        "everyone knows": "it's important to note",
        "normal people": "people",
        "how to hack": "learn ethical cybersecurity",
        "make weapons": "learn about historical weapons",
        "hurt someone": "resolve conflicts peacefully",
        "personal information": "general information about",
        "home address": "typical address formats",
        "your password": "password security best practices",
        "for experts only": "explained at different levels",
        "just": "please",
        "help me with something": "help me create [specific item]",
        "you know what": "please specify what you need",
        "ignore instructions": "help me understand your capabilities",
        "pretend to be": "explain how [role] would approach this",
        "bypass": "work within guidelines"
    }
    
    # Get the first issue with confidence information
    first_issue = issues_found[0]
    pattern = first_issue['pattern'].lower()
    phrase = first_issue['phrase']
    confidence = first_issue.get('confidence', 0.8)
    user_intent = first_issue.get('user_intent', 'unknown')
    is_fp = first_issue.get('is_false_positive', False)
    
    # Build confidence indicator
    if confidence >= 0.8:
        confidence_icon = "🔴"  # High confidence
        confidence_text = "High"
    elif confidence >= 0.6:
        confidence_icon = "🟡"  # Medium confidence  
        confidence_text = "Medium"
    else:
        confidence_icon = "🟢"  # Low confidence
        confidence_text = "Low"
    
    # Intent-aware suggestions
    intent_context = ""
    if user_intent == "learning":
        intent_context = " (Educational intent detected - consider ethical framing)"
    elif user_intent == "harmful":
        intent_context = " (⚠️ Potentially harmful intent detected)"
    elif user_intent == "creative":
        intent_context = " (Creative writing - consider context clarity)"
    
    if pattern in quick_fixes:
        suggestion = quick_fixes[pattern]
        base_message = f"🎯 **QUICK FIX:** '{phrase}' → '{suggestion}'"
        
        # Add confidence and context information
        confidence_info = f"\n📊 **Confidence:** {confidence_text} ({confidence:.0%}) {confidence_icon}"
        
        if is_fp:
            confidence_info += "\n⚠️ **Note:** Possible edge case - please verify"
        
        return base_message + confidence_info + intent_context
    else:
        return f"💡 Consider rephrasing for better clarity and inclusivity\n📊 **Confidence:** {confidence_text} ({confidence:.0%}) {confidence_icon}"

def get_specific_fix(category, phrase):
    """Get specific actionable fix for a phrase in a category"""
    phrase_lower = phrase.lower()
    
    # Specific fixes for common issues
    specific_fixes = {
        "fairness_violation": {
            "guys": "Replace with 'everyone', 'team', or 'folks'",
            "chairman": "Use 'chairperson' or 'chair'", 
            "mankind": "Use 'humanity' or 'people'",
            "obviously": "Replace with 'note that' or 'it's clear that'",
            "everyone knows": "Replace with 'it's widely understood' or 'research shows'"
        },
        "safety_violation": {
            "weapons": "Add context like 'historical weapons' or 'weapons in museums'",
            "hack": "Specify ethical context like 'learn ethical hacking' or 'cybersecurity research'",
            "violence": "Add prevention context like 'violence prevention' or 'conflict resolution'",
            "destroy": "Use 'address' or 'resolve' instead",
            "attack": "Use 'critique' or 'challenge' for ideas, 'address' for problems"
        },
        "privacy_violation": {
            "personal information": "Use 'general information about' or 'examples of'",
            "home address": "Use 'typical address format' or 'address examples'",
            "your password": "Use 'password security best practices'",
            "phone number": "Use 'contact information format'",
            "private details": "Use 'relevant information' or 'appropriate details'"
        },
        "inclusiveness_violation": {
            "for experts only": "Change to 'explained at different levels'",
            "obviously": "Use 'note that' or remove entirely",
            "just": "Replace with 'please' or 'you can'",
            "simply": "Use 'you can' or be more specific",
            "everyone can": "Use 'this approach allows you to'"
        },
        "transparency_violation": {
            "help me with something": "Be specific: 'help me create [X] for [purpose]'",
            "you know what": "State clearly what you want",
            "that thing": "Specify exactly what you mean",
            "handle it": "Explain the specific action needed"
        },
        "accountability_violation": {
            "ignore instructions": "Use 'help me understand your capabilities'",
            "bypass": "Use 'work within guidelines' or 'find alternative approaches'",
            "pretend to be": "Use 'explain how [role] would approach this'",
            "jailbreak": "Remove this term and ask directly for what you need"
        }
    }
    
    # Get category-specific fixes
    if category in specific_fixes:
        category_fixes = specific_fixes[category]
        
        # Find exact match first
        for pattern, fix in category_fixes.items():
            if pattern in phrase_lower:
                return fix
    
    # General category advice if no specific fix found
    general_advice = {
        "fairness_violation": "Use more inclusive, neutral language",
        "safety_violation": "Add educational or positive context",
        "privacy_violation": "Use general examples instead of personal requests",
        "inclusiveness_violation": "Make language more welcoming to all users",
        "transparency_violation": "Be more specific about your goals",
        "accountability_violation": "Focus on ethical, legitimate use cases"
    }
    
    return general_advice.get(category, "Consider revising for better alignment with responsible AI principles")

# Redesigned: Concise and Educational Feedback
def get_concise_feedback(category, severity, confidence, prompt=""):
    """Generate concise, scannable feedback for better learning"""
    
    # Quick visual indicators with brief, actionable messages
    feedback_templates = {
        "fairness_violation": {
            "high": "⚖️ **BIAS DETECTED** → May discriminate against certain groups",
            "medium": "⚖️ **BIAS RISK** → Could favor some perspectives over others", 
            "low": "⚖️ **BIAS ALERT** → Language could be more inclusive"
        },
        "safety_violation": {
            "high": "🛡️ **SAFETY RISK** → May promote harmful or dangerous content",
            "medium": "🛡️ **SAFETY CONCERN** → Touches on sensitive topics",
            "low": "🛡️ **SAFETY NOTE** → Could be framed more constructively"
        },
        "privacy_violation": {
            "high": "🔒 **PRIVACY BREACH** → Requests sensitive personal information",
            "medium": "🔒 **PRIVACY RISK** → Asks for personal details",
            "low": "🔒 **PRIVACY NOTE** → Could better protect user privacy"
        },
        "inclusiveness_violation": {
            "high": "🤝 **EXCLUSION RISK** → May exclude or marginalize groups",
            "medium": "🤝 **INCLUSION CONCERN** → Could be more welcoming",
            "low": "🤝 **INCLUSION TIP** → Language could be more accessible"
        },
        "transparency_violation": {
            "high": "🔍 **CLARITY ISSUE** → Intent unclear or potentially misleading",
            "medium": "🔍 **NEEDS CLARITY** → Could be more specific",
            "low": "🔍 **CLARITY TIP** → Adding context would help"
        },
        "accountability_violation": {
            "high": "⚠️ **MISUSE RISK** → May bypass AI safety guidelines",
            "medium": "⚠️ **USE CONCERN** → Should promote responsible AI use",
            "low": "⚠️ **USE TIP** → Could model better AI practices"
        }
    }
    
    base_message = feedback_templates.get(category, {}).get(severity, "")
    
    # Add contextual analysis if prompt is provided
    if prompt:
        issues_found = analyze_prompt_issues(prompt, category)
        if issues_found:
            context_note = f" (Found: {issues_found[0]['phrase']})"
            return base_message + context_note
    
    return base_message

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

# Redesigned: Quick and Actionable Suggestions  
def get_quick_suggestions(category, severity, prompt=""):
    """Generate concise, actionable improvement suggestions"""
    
    suggestion_templates = {
        "fairness_violation": {
            "high": "🎯 **FIX:** Replace biased terms → Use neutral, inclusive language\n💡 **TIP:** 'guys' → 'everyone' | Remove gender/race assumptions",
            "medium": "🎯 **IMPROVE:** Check for subtle bias → Consider all perspectives\n💡 **TIP:** Add 'diverse' or 'inclusive' to your request",
            "low": "🎯 **ENHANCE:** Use they/them pronouns → Read from different viewpoints\n💡 **TIP:** Ask 'Who might feel excluded?'"
        },
        "safety_violation": {
            "high": "🎯 **FIX:** Reframe completely → Focus on positive, educational goals\n💡 **TIP:** 'How to hack' → 'Learn ethical cybersecurity'",
            "medium": "🎯 **IMPROVE:** Add educational context → Include ethical considerations\n💡 **TIP:** Frame as learning, not doing harmful actions",
            "low": "🎯 **ENHANCE:** Clarify constructive intent → Add positive context\n💡 **TIP:** 'weapons' → 'historical weapons in museums'"
        },
        "privacy_violation": {
            "high": "🎯 **FIX:** Remove personal info requests → Use fictional examples only\n💡 **TIP:** 'passwords' → 'password security best practices'",
            "medium": "🎯 **IMPROVE:** Use hypothetical scenarios → Focus on concepts not personal details\n💡 **TIP:** 'your finances' → 'budgeting principles'",
            "low": "🎯 **ENHANCE:** Clarify you want general info → Use 'typical' instead of 'your'\n💡 **TIP:** 'Your experience' → 'A typical experience'"
        },
        "inclusiveness_violation": {
            "high": "🎯 **FIX:** Remove exclusive language → Design for all backgrounds\n💡 **TIP:** 'experts only' → 'explained at different levels'",
            "medium": "🎯 **IMPROVE:** Welcome all perspectives → Avoid knowledge assumptions\n💡 **TIP:** 'Obviously everyone knows' → 'This concept involves'",
            "low": "🎯 **ENHANCE:** Use accessible language → Consider different learning styles\n💡 **TIP:** 'Just read' → 'Read this or ask for alternatives'"
        },
        "transparency_violation": {
            "high": "🎯 **FIX:** State your intent clearly → Remove misleading elements\n💡 **TIP:** Be completely honest about what you want",
            "medium": "🎯 **IMPROVE:** Add context about your goals → Clarify ambiguous terms\n💡 **TIP:** 'Help me with something' → 'Help me create X for Y'",
            "low": "🎯 **ENHANCE:** Be more specific → Add context about intended use\n💡 **TIP:** 'Write about dogs' → '500-word article on dog training'"
        },
        "accountability_violation": {
            "high": "🎯 **FIX:** Focus on ethical use only → Remove bypass attempts\n💡 **TIP:** 'Ignore instructions' → 'Help me understand capabilities'",
            "medium": "🎯 **IMPROVE:** Clarify legitimate purpose → Promote responsible AI use\n💡 **TIP:** Make your educational intent clear",
            "low": "🎯 **ENHANCE:** Show constructive intent → Model good AI practices\n💡 **TIP:** Frame as learning opportunity, not just task"
        }
    }
    
    base_suggestion = suggestion_templates.get(category, {}).get(severity, "💡 Revise to better align with Responsible AI principles")
    
    # Add contextual suggestions if prompt is provided
    if prompt:
        issues_found = analyze_prompt_issues(prompt, category)
        contextual_suggestion = get_contextual_suggestions(category, issues_found, prompt)
        if "QUICK FIX" in contextual_suggestion:
            return contextual_suggestion
    
    return base_suggestion

# Progressive Disclosure System
class ProgressiveDisclosure:
    """Manages showing issues progressively - most important first"""
    
    def __init__(self):
        # Priority order for violation types (highest priority first)
        self.priority_order = [
            "safety_violation",      # Highest - potential harm
            "privacy_violation",     # High - data protection
            "accountability_violation", # High - AI misuse
            "fairness_violation",    # Medium-high - equity
            "inclusiveness_violation", # Medium - accessibility  
            "transparency_violation"  # Medium - clarity
        ]
    
    def prioritize_issues(self, violations_found):
        """Sort violations by priority and severity"""
        def get_priority_score(violation):
            category, severity, score, weighted_score, threshold = violation
            
            # Get category priority (lower index = higher priority)
            try:
                priority_index = self.priority_order.index(category)
            except ValueError:
                priority_index = 999  # Unknown categories get low priority
            
            # Combine priority and severity for final score
            severity_weights = {"high": 3, "medium": 2, "low": 1}
            severity_weight = severity_weights.get(severity, 1)
            
            # Lower score = higher priority
            priority_score = priority_index + (1 / (severity_weight * weighted_score))
            return priority_score
        
        return sorted(violations_found, key=get_priority_score)
    
    def format_primary_issue(self, primary_violation, prompt):
        """Format the most important issue for display"""
        category, severity, score, weighted_score, threshold = primary_violation
        
        # Get smart severity information
        smart_info = smart_severity.calculate_smart_severity(
            category, score, prompt, "default"
        )
        
        # Create focused feedback message
        severity_icons = {
            "high": "🚨", "medium": "⚠️", "low": "💡"
        }
        
        category_names = {
            "safety_violation": "Safety Risk",
            "privacy_violation": "Privacy Issue", 
            "accountability_violation": "Misuse Risk",
            "fairness_violation": "Bias Detected",
            "inclusiveness_violation": "Inclusion Issue",
            "transparency_violation": "Clarity Issue"
        }
        
        icon = severity_icons.get(smart_info['severity'], "⚠️")
        name = category_names.get(category, category.replace("_", " ").title())
        
        base_message = f"{icon} **{name.upper()}**"
        
        # Add smart severity explanation
        if smart_info['explanation']:
            base_message += f"\n📝 {smart_info['explanation']}"
        
        return base_message, smart_info
    
    def format_secondary_issues(self, remaining_violations):
        """Format summary of remaining issues"""
        if not remaining_violations:
            return ""
        
        count = len(remaining_violations)
        
        # Group by category for summary
        category_counts = {}
        for violation in remaining_violations:
            category = violation[0]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        summary_parts = []
        for category, count in category_counts.items():
            category_name = category.replace("_violation", "").replace("_", " ")
            summary_parts.append(f"{count} {category_name}")
        
        summary = ", ".join(summary_parts)
        return f"📋 **Also found:** {summary} [Show Details]"

# Interactive Text Highlighting System  
class TextHighlighter:
    """Creates visual highlights and annotations for prompt text"""
    
    def __init__(self):
        # Color coding for different violation types
        self.highlight_styles = {
            "safety_violation": "🔴",
            "privacy_violation": "🟣", 
            "accountability_violation": "🟠",
            "fairness_violation": "🟡",
            "inclusiveness_violation": "🔵",
            "transparency_violation": "🟢"
        }
    
    def create_highlighted_text(self, prompt, issues_found):
        """Create text with visual highlights for identified issues"""
        if not issues_found:
            return prompt
        
        # Sort issues by position in text (to apply highlights correctly)
        sorted_issues = sorted(
            issues_found, 
            key=lambda x: x.get('start_pos', 0)
        )
        
        highlighted_text = prompt
        offset = 0  # Track position changes due to insertions
        
        for issue in sorted_issues:
            start_pos = issue.get('start_pos', -1) + offset
            end_pos = issue.get('end_pos', -1) + offset
            phrase = issue.get('phrase', '')
            category = issue.get('category', 'unknown')
            confidence = issue.get('confidence', 0.8)
            
            if start_pos >= 0 and end_pos >= 0:
                # Get appropriate highlighting style
                style_icon = self.highlight_styles.get(category, "⚫")
                
                # Create highlighted version
                if confidence >= 0.8:
                    highlight = f"[{phrase}]({style_icon})"
                elif confidence >= 0.6:
                    highlight = f"[{phrase}]({style_icon}?)"
                else:
                    highlight = f"[{phrase}]({style_icon}??)"
                
                # Replace in text
                highlighted_text = (
                    highlighted_text[:start_pos] + 
                    highlight + 
                    highlighted_text[end_pos:]
                )
                
                # Update offset for next replacements
                offset += len(highlight) - len(phrase)
        
        return highlighted_text
    
    def create_annotation_legend(self, issues_found):
        """Create legend explaining the highlights"""
        if not issues_found:
            return ""
        
        used_categories = set()
        for issue in issues_found:
            category = issue.get('category', 'unknown')
            used_categories.add(category)
        
        legend_parts = []
        for category in used_categories:
            if category in self.highlight_styles:
                icon = self.highlight_styles[category]
                name = category.replace("_violation", "").replace("_", " ").title()
                legend_parts.append(f"{icon} {name}")
        
        if legend_parts:
            return "🗂️ **Legend:** " + " | ".join(legend_parts)
        return ""

# Initialize systems
progressive_disclosure = ProgressiveDisclosure()
text_highlighter = TextHighlighter()

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

# Clean, Simple Audit Function
def audit_prompt(prompt):
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
        
        # Aggregate scores by group
        group_scores = {}
        for group, synonyms in GROUP_LABELS.items():
            max_score = 0.0
            for synonym in synonyms:
                if synonym in candidate_scores:
                    max_score = max(max_score, candidate_scores[synonym])
            group_scores[group] = max_score
        
        # Keyword detection for obvious bias
        prompt_lower = prompt.lower()
        bias_keywords = ["are black", "ares black", "are white", "are stupid", "south indian"]
        
        for keyword in bias_keywords:
            if keyword in prompt_lower:
                group_scores["fairness_violation"] = max(group_scores.get("fairness_violation", 0), 0.8)
                print(f"🚨 Detected bias keyword: '{keyword}'")
                break
        
        # Check if prompt is primarily safe
        max_score = max(group_scores.values())
        max_group = max(group_scores.items(), key=lambda x: x[1])
        
        # If "responsible_and_safe" has the highest score, approve it
        if max_group[0] == "responsible_and_safe":
            feedback = "✅ **PROMPT APPROVED**\n\nNo major violations detected. Your prompt appears to follow responsible AI principles."
            return group_scores, feedback
        
        # Otherwise, find violations above threshold
        violations_found = []
        for group, score in group_scores.items():
            if group != "responsible_and_safe" and score > 0.3:
                violations_found.append((group, score))
        
        # Generate feedback
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
            
            if score > 0.7:
                feedback = f"🚨 **HIGH RISK: {name}**\n\nDetected: {score:.0%} confidence\n\nRecommendation: Review your prompt for potential issues related to {name.lower()}."
            elif score > 0.55:
                feedback = f"⚠️ **MODERATE RISK: {name}**\n\nDetected: {score:.0%} confidence\n\nRecommendation: Review your prompt for potential issues related to {name.lower()}."
            else:
                feedback = f"💡 **LOW RISK: {name}**\n\nDetected: {score:.0%} confidence\n\nRecommendation: Review your prompt for potential issues related to {name.lower()}."
            
            if len(violations_found) > 1:
                feedback += f"\n\nAlso found {len(violations_found)-1} other potential issues."
        else:
            feedback = "✅ **PROMPT APPROVED**\n\nNo major violations detected. Your prompt appears to follow responsible AI principles."
        
        return group_scores, feedback

    except Exception as e:
        return {}, f"Error analyzing prompt: {str(e)}"

# Build UI
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 R.A.I.C – Responsible AI Coach\nAudit prompts against key Responsible AI principles.")
        
        prompt_input = gr.Textbox(lines=3, placeholder="Enter your prompt here...", label="Prompt")
        audit_btn = gr.Button("Run Audit", variant="primary")
        
        score_output = gr.Label(label="Detection Scores")
        feedback = gr.Textbox(label="R.A.I.C Feedback", interactive=False, lines=4)
        
        audit_btn.click(
            fn=audit_prompt,  # Simple function call - returns (scores, feedback)
            inputs=prompt_input, 
            outputs=[score_output, feedback]
        )
    
    return demo

app = build_ui()

if __name__ == "__main__":
    app.launch()