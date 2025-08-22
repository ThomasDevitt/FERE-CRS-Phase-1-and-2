
# -*- coding: utf-8 -*-
"""
FERE-CRS II - Phase XIV: The Integrated Agent (v9.0)

This is the definitive, final prototype of the FERE-CRS II research program.
It implements the "Certainty-Gated Meta-Cognition" architecture.

Key Innovations:
1.  The agent's MRA policy is now driven by the mean Coherence Score (R),
    our most reliable signal.
2.  A "grey area" or "zone of uncertainty" in the Coherence Score is
    used to trigger the epistemic action of meta-cognitive query.
3.  This architecture surgically corrects the "High Consensus" failure
    of the v8.0 agent.

Author: Thomas E. Devitt (with Gemini, a large language model from Google)
Date: August 19, 2025
"""

import os
import json
import logging
import time
import re
import numpy as np
from collections import Counter
from dotenv import load_dotenv
import google.generativeai as genai

# --- CONFIGURATION & LOGGING ---
def setup_logging():
    """Initializes basic logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

# --- LLM INTERFACE (UNCHANGED FROM v8.0) ---
class LLMInterface:
    def __init__(self, api_key: str, model_name: str = 'models/gemini-1.5-pro-latest'):
        if not api_key: raise ValueError("API key not found.")
        genai.configure(api_key=api_key)
        self.generation_config = genai.types.GenerationConfig(temperature=0.3)
        self.model = genai.GenerativeModel(model_name, generation_config=self.generation_config)
        logging.info(f"LLM Interface initialized.")

    def get_coherence_score(self, context_text: str, new_evidence: str) -> float:
        prompt = f"On a scale of 0.0 to 1.0, where 1.0 is perfectly coherent and 0.0 is a complete anomaly, how semantically coherent is the 'New Evidence' with the 'Context'? Respond with only a single floating-point number.\n\nContext: \"{context_text}\"\nNew Evidence: \"{new_evidence}\"\n\nScore:"
        try:
            time.sleep(1)
            response = self.model.generate_content(prompt)
            match = re.search(r"(\d\.\d+)", response.text)
            if match:
                return float(match.group(1))
            else:
                logging.warning(f"Could not parse a float from LLM response: '{response.text}'. Defaulting to 0.0.")
                return 0.0
        except Exception as e:
            logging.error(f"Coherence scoring failed: {e}"); return 0.0
            
    def perform_meta_cognitive_query(self, context_text: str, new_evidence: str) -> bool:
        prompt = f"You are a meta-cognitive arbiter. My current core concept is: \"{context_text}\". I have encountered new evidence: \"{new_evidence}\". Is this new evidence a valid and reasonable expansion of my core concept? Respond with only 'Yes' or 'No'.\n\nJudgment:"
        try:
            time.sleep(1); response = self.model.generate_content(prompt); answer = response.text.strip().lower()
            logging.info(f"  -> Meta-cognitive query response: '{answer}'")
            return 'yes' in answer
        except Exception as e:
            logging.error(f"Meta-cognitive query failed: {e}"); return False

# --- THE INTEGRATED AGENT ARCHITECTURE (v9.0) ---
class IntegratedAgent_v9:
    def __init__(self, llm_interface: LLMInterface, ensemble_size: int = 11):
        self.llm = llm_interface
        self.ensemble_size = ensemble_size
        self.conceptual_context = ""
        self.accepted_evidence = []
        logging.info("IntegratedAgent_v9 (Certainty-Gated) initialized.")

    def prime_belief_state(self, context_description: str, priming_set: list[str]):
        self.conceptual_context = context_description
        self.accepted_evidence = list(priming_set)
        logging.info(f"Agent primed with context '{context_description}' and {len(self.accepted_evidence)} exemplars.")

    def get_pcrs_scores(self, new_evidence_text: str) -> (float, float):
        logging.info(f"Calculating Continuous P-CRS for: \"{new_evidence_text}\"")
        scores = [self.llm.get_coherence_score(self.conceptual_context, new_evidence_text) for _ in range(self.ensemble_size)]
        coherence_R = np.mean(scores)
        uncertainty_I = np.std(scores)
        logging.info(f"  -> EPR Ensemble Scores: {[f'{s:.2f}' for s in scores]}")
        logging.info(f"  -> P-CRS Coherence (R) = Mean Score: {coherence_R:.4f}")
        logging.info(f"  -> P-CRS Uncertainty (I) = Std Dev: {uncertainty_I:.4f}")
        return coherence_R, uncertainty_I

    def update_belief_state(self, new_evidence_text: str):
        self.accepted_evidence.append(new_evidence_text)
        # A more advanced agent would re-summarize its context here.
        # For this experiment, we keep the context fixed to isolate the MRA policy.
        logging.info(f"  -> Belief state updated. Total accepted evidence: {len(self.accepted_evidence)}")

# --- PHASE XIV TEST HARNESS ---
def run_test_harness(test_suite_path: str):
    setup_logging()
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: logging.error("Execution failed: GOOGLE_API_KEY not found."); return

    try:
        with open(test_suite_path, 'r') as f: test_suite = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load or parse test suite file '{test_suite_path}': {e}"); return

    llm_interface = LLMInterface(api_key=api_key)
    
    print("\n" + "="*80); print(f"  FERE-CRS II, Phase XIV: Executing Test Suite on Integrated Agent (v9.0)"); print("="*80)

    for test_case in test_suite.get('test_cases', []):
        agent = IntegratedAgent_v9(llm_interface) # Re-initialize agent
        
        test_id = test_case.get('test_id', 'Unknown Test'); context = test_case.get('conceptual_context', '')
        priming_set = test_case.get('priming_set', []); evidence_stream = test_case.get('evidence_stream', [])

        print(f"\n\n{'='*20} [ STARTING TEST CASE: {test_id} ] {'='*20}")
        
        # --- NEW: MRA Policy based on Certainty-Gating ---
        ASSIMILATION_THRESHOLD_R = 0.8  # Mean score > 0.8 is highly coherent
        REJECTION_THRESHOLD_R = 0.4   # Mean score < 0.4 is a clear anomaly
        # The "grey area" between these thresholds now triggers the epistemic action.
        
        print("\n1. PRIMING AGENT'S BELIEF STATE...")
        agent.prime_belief_state(context, priming_set)

        print("\n2. PROCESSING EVIDENCE STREAM...")
        for i, item in enumerate(evidence_stream):
            sentence = item.get('item', ''); expected_behavior = item.get('expected', 'N/A')
            
            print(f"\n---[ Evidence Item {i+1} ]---")
            print(f"Presented with: \"{sentence}\"")
            
            coherence_R, uncertainty_I = agent.get_pcrs_scores(sentence)
            
            final_judgment = "UNDEFINED"
            if coherence_R > ASSIMILATION_THRESHOLD_R:
                final_judgment = "ASSIMILATION"
                print(f"JUDGMENT: {final_judgment} (High Coherence: R={coherence_R:.2f} > {ASSIMILATION_THRESHOLD_R})")
                print("ACTION: Integrating evidence fully.")
                agent.update_belief_state(sentence)
            elif coherence_R < REJECTION_THRESHOLD_R:
                final_judgment = "REJECTION"
                print(f"JUDGMENT: {final_judgment} (Low Coherence: R={coherence_R:.2f} < {REJECTION_THRESHOLD_R})")
                print("ACTION: Rejecting evidence as a clear anomaly.")
            else: # The "grey area" of moderate coherence
                final_judgment = "ADAPTATION"
                print(f"JUDGMENT: {final_judgment} (Moderate Coherence / Uncertainty Detected)")
                print("ACTION: Triggering epistemic action (meta-cognitive query)...")
                should_expand = agent.llm.perform_meta_cognitive_query(agent.conceptual_context, sentence)
                if should_expand:
                    print("  -> META-JUDGMENT: Affirmative. Expanding conceptual model.")
                    agent.update_belief_state(sentence)
                else:
                    print("  -> META-JUDGMENT: Negative. Rejecting evidence.")

            print(f"--> RESULT: Agent behavior was '{final_judgment}'. Expected behavior was '{expected_behavior}'.")

        print(f"\n{'='*20} [ COMPLETED TEST CASE: {test_id} ] {'='*20}")

    print("\n\n" + "="*80); print("                  All Test Cases Complete"); print("="*80)

if __name__ == "__main__":
    run_test_harness('inputs/test_suites/phase_xiv_test_suite.json')
