import os, json, csv, time, requests, random
from pathlib import Path

# ===============================
# CONFIG
# ===============================
MODEL_NAME = "gpt-5.4-mini"
BASE_DIR = "./generator_outputs"   # directory holding the three generator result JSONs

INPUT_JSONS = {
    "Llama-3.1-8B": f"{BASE_DIR}/meta-llama_Llama-3.1-8B-Instruct.json",
    "Llama-3.3-70B": f"{BASE_DIR}/meta-llama_Llama-3.3-70B-Instruct.json",
    "HuatuoGPT-o1-8B": f"{BASE_DIR}/FreedomIntelligence_HuatuoGPT-o1-8B.json",
    
}


SLEEP_SEC = 0.5 
MAX_RETRIES = 5
BACKOFF_BASE_SEC = 2.0  
RESUME = True

# ===============================
# PROMPT BUILDER
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

# ===============================
# VERIFY FUNCTION
# ===============================
def verify(text: str):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": build_verification_prompt(text)}],
        "temperature": 0.0,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                print(f"[WARN] HTTP {r.status_code} retry {attempt}/{MAX_RETRIES} in {wait:.1f}s")
                time.sleep(wait)
                continue

            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]

            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end < start:
                return "INVALID", 0.0

            j = json.loads(content[start:end + 1])
            return j.get("label", "INVALID"), float(j.get("score", 0.0))

        except Exception as e:
            wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            print(f"[ERROR] Verification failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(wait)

    return "ERROR", 0.0

# ===============================
# DATA LOADING (전체 데이터 추출)
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
    if not Path(csv_path).exists():
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # generator_model, case_id, sample_idx 조합을 키로 사용
            done.add((row["generator_model"], row["case_id"], str(row["sample_idx"])))
    return done

# ===============================
# MAIN EXECUTION
# ===============================
def main():
    if not os.environ.get('OPENAI_API_KEY'):
        print("❌ ERROR: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    for model_name, json_path in INPUT_JSONS.items():
        out_csv = f"{BASE_DIR}/verification_{MODEL_NAME}_{model_name}.csv"

        print(f"\n[START] {model_name} | Loading ALL samples from {json_path}...")
        samples = load_all_samples(json_path, model_name)
        total_samples = len(samples)
        
        if total_samples == 0:
            continue
            
        print(f"[OK] Total {total_samples} samples identified.")

        done_keys = load_done_keys(out_csv) if RESUME else set()
        if RESUME and done_keys:
            print(f"[RESUME] Found {len(done_keys)} existing records. Skipping...")

        file_exists = Path(out_csv).exists()
        
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["generator_model", "case_id", "sample_idx", "label", "score"])

            written_in_this_run = 0
            for i, s in enumerate(samples, 1):
                key = (s["generator_model"], s["case_id"], str(s["sample_idx"]))
                
                if RESUME and key in done_keys:
                    continue

                label, score = verify(s["text"])
                writer.writerow([s["generator_model"], s["case_id"], s["sample_idx"], label, score])
                f.flush() # 실시간으로 파일에 기록 보장
                
                written_in_this_run += 1
                print(f"[{model_name}] Progress: {i}/{total_samples} | Result: {label} ({score})")
                
                time.sleep(SLEEP_SEC)

        print(f"[DONE] {model_name} processing complete. Result saved to: {out_csv}")

    print("\n🎉 ALL MODELS COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()