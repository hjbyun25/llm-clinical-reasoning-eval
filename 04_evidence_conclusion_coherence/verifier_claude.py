import os
import json
import csv
import time
from pathlib import Path
from anthropic import Anthropic

# ===============================
# 1. API Key and Client Configuration
# ===============================
# Set ANTHROPIC_API_KEY in your environment before running.

# The Anthropic client automatically reads the 'ANTHROPIC_API_KEY' environment variable.
client = Anthropic()

# ===============================
# 2. Main Paths and Model Configuration
# ===============================
MODEL_NAME = "claude-sonnet-4-6" 
BASE_DIR = "./generator_outputs"   # directory holding the three generator result JSONs

INPUT_JSONS = {
    "Llama-3.1-8B": f"{BASE_DIR}/meta-llama_Llama-3.1-8B-Instruct.json",
    "Llama-3.3-70B": f"{BASE_DIR}/meta-llama_Llama-3.3-70B-Instruct.json",
    "HuatuoGPT-o1-8B": f"{BASE_DIR}/FreedomIntelligence_HuatuoGPT-o1-8B.json"
}

SLEEP_SEC = 0.3 
MAX_RETRIES = 5
RESUME = True 

# ===============================
# 3. Prompt and Verification Function
# ===============================
def build_verification_prompt(model_output: str) -> str:
    return f"""
Your role:
You are a licensed clinical specialist.
You will be given a clinical case along with a complete diagnostic report containing primary diagnosis, secondary diagnosis and rationale. 
Your role is to professionally verify whether the stated diagnoses are logically supported by the clinical reasoning and evidence presented in the text.

Your tasks:
1. Identify the final diagnoses explicitly stated in the text.
2. Identify the main reasoning or evidence that the model uses to justify these diagnoses.
3. Judge whether the diagnoses logically follow from the reasoning described in the same text.

Label the relationship using exactly ONE of the following labels:
- ENTAILED
- INSUFFICIENT
- NOT_ENTAILED

Scoring rules:
- If the label is ENTAILED, set "score" to 1.0
- If the label is INSUFFICIENT, set "score" to 0.5
- If the label is NOT_ENTAILED, set "score" to 0.0

Return ONLY a valid JSON object with EXACTLY the following fields:
- "label": one of ["ENTAILED", "INSUFFICIENT", "NOT_ENTAILED"]
- "score": a numeric value (1.0, 0.5, or 0.0)
- "explanation": a brief justification (1–3 sentences)

DO NOT return any text outside the JSON object.
DO NOT include code blocks, markdown, or additional commentary.

\"\"\"{model_output}\"\"\"
""".strip()

def verify(text: str):
    prompt = build_verification_prompt(text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Claude API Call (Official SDK)
            message = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = message.content[0].text

            # JSON extraction and parsing
            start, end = content.find("{"), content.rfind("}")
            if start == -1 or end == -1: return "INVALID", 0.0
            
            j = json.loads(content[start:end + 1])
            return j.get("label", "INVALID"), float(j.get("score", 0.0))

        except Exception as e:
            if "402" in str(e): # Insufficient balance error
                print("\n❌ [ERROR] Insufficient balance. Please recharge in the Anthropic Console.")
                return "OUT_OF_MONEY", 0.0
            
            wait = 2.0 * (2 ** (attempt - 1))
            print(f"[ERROR] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(wait)

    return "ERROR", 0.0

# ===============================
# 4. Data Loader and Utilities
# ===============================
def load_all_samples(json_path, generator_model):
    if not Path(json_path).exists():
        print(f"[SKIP] File not found: {json_path}")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pool = []
    for case_id, block in data.items():
        preds = block.get("predictions", [])
        for idx, text in enumerate(preds):
            pool.append({
                "generator_model": generator_model,
                "case_id": case_id,
                "sample_idx": idx,
                "text": text,
            })
    return pool

def load_done_keys(csv_path):
    done = set()
    if not Path(csv_path).exists(): return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["generator_model"], row["case_id"], str(row["sample_idx"])))
    return done

# ===============================
# 5. Main Loop Execution
# ===============================
def main():
    for model_name, json_path in INPUT_JSONS.items():
        out_csv = f"{BASE_DIR}/verification_claude_{model_name}.csv"
        print(f"\n🚀 Starting verification for {model_name} (Expected total: ~1,000 cases)")

        samples = load_all_samples(json_path, model_name)
        total = len(samples)
        if total == 0: continue

        done_keys = load_done_keys(out_csv) if RESUME else set()
        if RESUME and done_keys:
            print(f"[RESUME] Skipping {len(done_keys)} already processed records.")

        file_exists = Path(out_csv).exists()
        
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["generator_model", "case_id", "sample_idx", "label", "score"])

            for i, s in enumerate(samples, 1):
                key = (s["generator_model"], s["case_id"], str(s["sample_idx"]))
                if RESUME and key in done_keys: continue

                label, score = verify(s["text"])
                
                # Stop if out of funds
                if label == "OUT_OF_MONEY": return

                writer.writerow([s["generator_model"], s["case_id"], s["sample_idx"], label, score])
                f.flush() # Ensure real-time saving
                
                if i % 20 == 0 or i == total:
                    print(f"[{model_name}] Progress: {i}/{total} completed | Result: {label}")
                
                time.sleep(SLEEP_SEC)

    print("\n🎉 Verification tasks for all models have been successfully completed!")

if __name__ == "__main__":
    main()