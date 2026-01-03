import joblib
import os
import numpy as np

class RiskEngine:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.loaded = False
        
        try:
            # Load all layers
            self.vec_risk = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
            self.model_risk = joblib.load(os.path.join(models_dir, "risk_model_lr.joblib"))
            
            self.vec_intent = joblib.load(os.path.join(models_dir, "tfidf_vectorizer_intent.joblib"))
            self.model_intent = joblib.load(os.path.join(models_dir, "intent_classifier.joblib"))
            
            self.vec_issue = joblib.load(os.path.join(models_dir, "tfidf_vectorizer_issue.joblib"))
            self.model_issue = joblib.load(os.path.join(models_dir, "issue_classifier.joblib"))
            
            self.loaded = True
            print("✅ Risk Engine Loaded.")
            
        except FileNotFoundError:
            print(f"❌ CRITICAL ERROR: Models not found in '{models_dir}'.")
            self.loaded = False

    def predict(self, text, mode="balanced"):
        if not self.loaded: return {"decision": "ERROR", "reason": "Models Offline"}
        
        clean_text = str(text).lower().strip()
        
        # --- LAYER 1: BASE RISK (The Probability) ---
        vec_r = self.vec_risk.transform([clean_text])
        base_safe_prob = self.model_risk.predict_proba(vec_r)[0][1]
        base_risk = 1.0 - base_safe_prob
        
        # --- LAYER 2: INTENT (Context) ---
        vec_i = self.vec_intent.transform([clean_text])
        intent_label = self.model_intent.predict(vec_i)[0]
        
        # --- LAYER 3: ISSUE CLASSIFICATION (With Null Confidence Logic) ---
        vec_s = self.vec_issue.transform([clean_text])
        issue_probs = self.model_issue.predict_proba(vec_s)[0]
        max_issue_conf = np.max(issue_probs)
        raw_issue_label = self.model_issue.classes_[np.argmax(issue_probs)]
        
        # LOGIC CHANGE 1: Null Confidence Fallback
        # If the model is confused (low confidence), assume GENERAL_SUPPORT
        # This prevents "Praise" being misclassified as "Bugs"
        CONFIDENCE_THRESHOLD = 0.50
        if max_issue_conf < CONFIDENCE_THRESHOLD:
            issue_label = "GENERAL_SUPPORT"
        else:
            issue_label = raw_issue_label

        # --- LOGIC CHANGE 2: POLICY WEIGHTS ---
        # Instead of just "Veto", we add risk based on the severity of the issue.
        # This allows for nuanced decisions.
        RISK_WEIGHTS = {
            "DATA_LOSS": 0.5,          # Catastrophic
            "ACCOUNT_ACCESS": 0.5,     # Security Critical
            "PAYMENT_PROBLEM": 0.4,    # High Financial
            "SOFTWARE_BUG": 0.3,       # Medium
            "HARDWARE_FAILURE": 0.3,   # Medium
            "CONNECTIVITY_ISSUE": 0.2, # Low-Medium
            "DELIVERY_PROBLEM": 0.2,   # Low-Medium
            "GENERAL_SUPPORT": 0.0     # Low
        }
        
        # Fallback for unknown labels
        policy_risk_adder = RISK_WEIGHTS.get(issue_label, 0.1)
        
        # CALCULATE COMPOSITE RISK
        # Base Risk + Policy Weight
        # e.g., 0.1 (Safe Text) + 0.5 (Data Loss) = 0.6 (Unsafe)
        final_risk_score = base_risk + policy_risk_adder
        
        # Cap at 1.0
        final_risk_score = min(1.0, final_risk_score)
        final_safe_prob = 1.0 - final_risk_score

        # --- DECISION THRESHOLDS ---
        # We adjust the acceptance threshold based on Mode
        thresholds = {
            "conservative": 0.85, # Needs 85% safety (Very hard to automate)
            "balanced": 0.65,     # Needs 65% safety
            "aggressive": 0.50    # Needs 50% safety
        }
        required_safety = thresholds.get(mode, 0.65)
        
        is_safe = final_safe_prob >= required_safety
        
        reason = f"Risk Score: {final_risk_score:.2f} (Base {base_risk:.2f} + {issue_label} {policy_risk_adder:.2f})"
        if not is_safe:
            if policy_risk_adder >= 0.4:
                reason = f"Policy Veto: {issue_label} (High Risk)"
            else:
                reason = f"Confidence too low ({final_safe_prob:.2f} < {required_safety})"

        return {
            "text": text,
            "decision": "AUTOMATE" if is_safe else "HUMAN_REVIEW",
            "reason": reason,
            "metrics": {
                "safe_prob": round(final_safe_prob, 4),
                "risk_score": round(final_risk_score, 4),
                "base_risk": round(base_risk, 4),
                "issue_conf": round(float(max_issue_conf), 4)
            },
            "signals": {
                "intent": intent_label,
                "issue": issue_label
            }
        }