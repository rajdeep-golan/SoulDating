# Minimal stand-in (original file was not in the repo): returns the logged conversation
import json, os
def extract_biodata(user_id):
    p = os.path.join("logs", f"{user_id}.jsonl")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {"messages": [json.loads(l) for l in f]}
