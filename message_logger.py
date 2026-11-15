# Minimal stand-in (original file was not in the repo): appends chat lines to logs/<user>.jsonl
import json, os, time
def log_message(user_id, role, content):
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", f"{user_id}.jsonl"), "a") as f:
        f.write(json.dumps({"t": time.time(), "role": role, "content": str(content)}) + "\n")
