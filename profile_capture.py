"""Turns a finished Zoey call into a structured dating profile and stores it.

1. Builds the transcript from the agent session history.
2. Asks Groq (JSON mode) to extract a profile using a fixed schema/taxonomy.
3. Saves a local backup in logs/profiles/ and POSTs it to the website DB
   (https://www.zoeyai.love/api/profiles), which also computes match percentages.
"""
import json
import logging
import os
import time
import uuid

import requests

logger = logging.getLogger("profile_capture")

INTERESTS = [
    "music", "travel", "hiking", "outdoors", "cooking", "food", "fitness", "yoga", "reading",
    "movies", "tv", "gaming", "art", "dance", "sports", "tech", "photography", "pets",
    "nightlife", "fashion", "spirituality", "volunteering", "coffee", "writing", "nature",
]
VALUES = [
    "family", "honesty", "loyalty", "respect", "ambition", "adventure", "kindness", "humor",
    "faith", "independence", "growth", "creativity",
]

SCHEMA_PROMPT = f"""
You extract a dating profile from a voice conversation between Zoey (a matchmaking AI) and a user.
Infer gender from words like guy/man/boy (male) or girl/woman/lady (female); if they say they want a girl/woman, looking_for is female (and vice versa).
Map synonyms to tags, e.g. trust/truthful -> honesty, faithful -> loyalty, funny -> humor, music/guitar/singing -> music, gym/working out -> fitness.
Return ONLY a JSON object with exactly these keys (use null when unknown, never invent facts):
{{
  "name": string|null,
  "age": integer|null,
  "gender": "male"|"female"|"non-binary"|null,
  "looking_for": "male"|"female"|"any"|null,
  "hometown": string|null,
  "current_city": string|null,
  "dream_city": string|null,
  "height_cm": integer|null,            // convert feet/inches to cm
  "occupation": string|null,
  "interests": string[],                // ONLY tags from: {INTERESTS}
  "likes_raw": string[],                // the user's own words
  "dislikes": string[],                 // ONLY tags from the interests list above (things they dislike)
  "dislikes_raw": string[],
  "values": string[],                   // ONLY tags from: {VALUES} (infer gently from what they said)
  "relationship_goal": "long-term"|"marriage"|"casual"|"not sure"|null,
  "personality": "introvert"|"extrovert"|"ambivert"|null,
  "summary": string                     // 1-2 warm sentences describing the person
}}
"""


def transcript_from_history(history) -> list[dict]:
    lines = []
    for item in getattr(history, "items", []):
        role = getattr(item, "role", None)
        if role not in ("user", "assistant"):
            continue
        text = getattr(item, "text_content", None)
        if callable(text):
            text = text()
        if text:
            lines.append({"role": "user" if role == "user" else "zoey", "text": text})
    return lines


def extract_profile(transcript: list[dict]) -> dict:
    convo = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in transcript)
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={
            "model": "openai/gpt-oss-120b",
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SCHEMA_PROMPT},
                {"role": "user", "content": convo},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    profile = json.loads(resp.json()["choices"][0]["message"]["content"])
    profile["interests"] = [t for t in profile.get("interests") or [] if t in INTERESTS]
    profile["dislikes"] = [t for t in profile.get("dislikes") or [] if t in INTERESTS]
    profile["values"] = [t for t in profile.get("values") or [] if t in VALUES]
    return profile


def save_call_profile(transcript: list[dict], room_name: str) -> dict | None:
    if not any(t["role"] == "user" for t in transcript):
        logger.info("no user speech in %s, skipping profile", room_name)
        return None
    try:
        profile = extract_profile(transcript)
    except Exception:
        logger.exception("profile extraction failed")
        profile = {"summary": "Extraction failed; transcript saved.", "interests": [], "dislikes": [], "values": []}

    doc = {
        "id": f"call-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        "source": "voice_call",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "room": room_name,
        **profile,
        "transcript": transcript,
    }

    try:
        os.makedirs("logs/profiles", exist_ok=True)
        with open(f"logs/profiles/{doc['id']}.json", "w") as f:
            json.dump(doc, f, indent=2)
    except Exception:
        logger.exception("local profile copy failed; still sending to DB")

    url, secret = os.getenv("ZOEY_INGEST_URL"), os.getenv("ZOEY_INGEST_SECRET")
    if url and secret:
        try:
            r = requests.post(url, json=doc, headers={"x-ingest-secret": secret}, timeout=30)
            logger.info("profile %s sent to DB: %s %s", doc["id"], r.status_code, r.text[:200])
        except Exception:
            logger.exception("failed to send profile to DB")
    return doc
