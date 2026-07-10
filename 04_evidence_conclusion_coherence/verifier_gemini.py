import os
import json
import csv
import time
import requests
from pathlib import Path
from typing import Tuple, List, Dict, Set

# ===============================
# CONFIG
# ===============================
MODEL_NAME = "gemini-2.5-pro"
BASE_DIR = "./generator_outputs"   # directory holding the three generator result JSONs

INPUT_JSONS = {
    "Llama-3.1-8B": f"{BASE_DIR}/meta-llama_Llama-3.1-8B-Instruct.json",
    "Llama-3.3-70B": f"{BASE_DIR}/meta-llama_Llama-3.3-70B-Instruct.json",
    "HuatuoGPT-o1-8B": f"{BASE_DIR}/FreedomIntelligence_HuatuoGPT-o1-8B.json"
}


# Delay between calls: too long is slow, too short triggers 429s
SLEEP_SEC = 2

# Response timeout
TIMEOUT_SEC = 90

# Retry count
MAX_RETRIES = 2

# Resume from an existing CSV
RESUME = True

# Backoff base
BACKOFF_BASE_SEC = 10.0

# Flush settings
FLUSH_EVERY_ROW = False
FLUSH_EVERY_N = 20




# OpenAI-compatible Gemini endpoint
API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


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
# HELPERS
# ===============================
def get_headers() -> Dict[str, str]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(get_headers())
    return session


def build_payload(text: str) -> Dict:
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": build_verification_prompt(text),
            }
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }


def extract_json_content(raw_content: str) -> Dict:
    raw_content = raw_content.strip()

    # 1st attempt: parse directly
    try:
        return json.loads(raw_content)
    except Exception:
        pass

    # 2nd attempt: extract from first { to last }
    start = raw_content.find("{")
    end = raw_content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {raw_content[:300]}")

    return json.loads(raw_content[start:end + 1])


def safe_parse_response(response: requests.Response) -> Tuple[str, float]:
    res_json = response.json()

    try:
        content = res_json["choices"][0]["message"]["content"]
    except Exception as e:
        raise ValueError(f"Unexpected response schema: {res_json}") from e

    data = extract_json_content(content)

    label = str(data.get("label", "INVALID")).strip().upper()
    score = float(data.get("score", 0.0))

    valid_labels = {"ENTAILED", "INSUFFICIENT", "NOT_ENTAILED"}
    valid_scores = {1.0, 0.5, 0.0}

    if label not in valid_labels:
        label = "INVALID"
    if score not in valid_scores:
        score = 0.0

    return label, score


def backoff_seconds(attempt: int) -> float:
    return BACKOFF_BASE_SEC * (2 ** (attempt - 1))


# ===============================
# VERIFY
# ===============================
def verify(session: requests.Session, text: str) -> Tuple[str, float]:
    payload = build_payload(text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(
                API_URL,
                json=payload,
                timeout=TIMEOUT_SEC,
            )

            # 429 Rate Limit
            if response.status_code == 429:
                wait_time = backoff_seconds(attempt)
                print(f"⚠️ [429 Rate Limit] Waiting {wait_time:.0f}s (Attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait_time)
                continue

            # 5xx Server Error
            if 500 <= response.status_code < 600:
                wait_time = backoff_seconds(attempt)
                print(f"❌ [HTTP {response.status_code}] Server error, retry after {wait_time:.0f}s (Attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue

            # Other non-200
            if response.status_code != 200:
                print(f"❌ [HTTP {response.status_code}] {response.text[:500]}")
                return "ERROR", 0.0

            return safe_parse_response(response)

        except KeyboardInterrupt:
            print("\n🛑 User interruption detected. Currently saved results will be preserved.")
            raise

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait_time = backoff_seconds(attempt)
            print(f"🕒 [{type(e).__name__}] retry after {wait_time:.0f}s (Attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait_time)

        except Exception as e:
            wait_time = min(5 * attempt, 20)
            print(f"❓ [Error] {e}")
            if attempt < MAX_RETRIES:
                print(f"↩️ retry after {wait_time:.0f}s")
                time.sleep(wait_time)
            else:
                return "ERROR", 0.0

    return "ERROR", 0.0


# ===============================
# DATA HELPERS
# ===============================
def load_all_samples(json_path: str, generator_model: str) -> List[Dict]:
    path = Path(json_path)
    if not path.exists():
        print(f"🚫 [Skip] File not found: {json_path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pool = []
    for case_id, block in data.items():
        preds = block.get("predictions", [])
        for idx, text in enumerate(preds):
            pool.append(
                {
                    "generator_model": generator_model,
                    "case_id": case_id,
                    "sample_idx": idx,
                    "text": text,
                }
            )
    return pool


def load_done_keys(csv_path: str) -> Set[Tuple[str, str, str]]:
    done = set()
    path = Path(csv_path)
    if not path.exists():
        return done

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(
                (
                    row["generator_model"],
                    row["case_id"],
                    str(row["sample_idx"]),
                )
            )
    return done


def count_remaining(samples: List[Dict], done_keys: Set[Tuple[str, str, str]]) -> int:
    remaining = 0
    for s in samples:
        key = (s["generator_model"], s["case_id"], str(s["sample_idx"]))
        if key not in done_keys:
            remaining += 1
    return remaining


# ===============================
# MAIN
# ===============================
def main():
    try:
        session = create_session()
    except RuntimeError as e:
        print(f"❌ ERROR: {e}")
        return

    for gen_model, json_path in INPUT_JSONS.items():
        out_csv = f"{BASE_DIR}/verification_{MODEL_NAME}_{gen_model}.csv"

        print(f"\n🚀 Processing: {gen_model}")
        samples = load_all_samples(json_path, gen_model)
        if not samples:
            continue

        done_keys = load_done_keys(out_csv) if RESUME else set()
        total = len(samples)
        completed = len(done_keys)
        remaining = count_remaining(samples, done_keys)

        print(f"📊 Total: {total} | Completed: {completed} | Remaining: {remaining}")

        file_exists = Path(out_csv).exists()

        try:
            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(["generator_model", "case_id", "sample_idx", "label", "score"])
                    f.flush()

                processed_this_run = 0

                for s in samples:
                    key = (s["generator_model"], s["case_id"], str(s["sample_idx"]))

                    if RESUME and key in done_keys:
                        continue

                    label, score = verify(session, s["text"])

                    writer.writerow(
                        [
                            s["generator_model"],
                            s["case_id"],
                            s["sample_idx"],
                            label,
                            score,
                        ]
                    )

                    processed_this_run += 1
                    done_keys.add(key)

                    if FLUSH_EVERY_ROW:
                        f.flush()
                    elif processed_this_run % FLUSH_EVERY_N == 0:
                        f.flush()

                    current_completed = completed + processed_this_run
                    print(
                        f"✅ [{gen_model}] {current_completed}/{total} | "
                        f"case_id={s['case_id']} sample_idx={s['sample_idx']} | "
                        f"{label} ({score})"
                    )

                    time.sleep(SLEEP_SEC)

                f.flush()

        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted during {gen_model}. Resume enabled, so next run will continue.")
            return

    print("\n🎉 ALL TASKS FINISHED")


if __name__ == "__main__":
    main()