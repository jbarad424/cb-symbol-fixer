"""
CB Creative Studio — Remote Symbol Fixer
=========================================
Post-processing module that detects and corrects Chubby Buttons remote
symbols in AI-generated marketing images.

Pipeline:
  1. Claude Vision locates the remote and assesses symbol accuracy
  2. FLUX Kontext Pro fixes symbols via instruction-based editing
  3. Claude Vision QA checks the result
  4. Retries up to MAX_ATTEMPTS if QA fails

Environment variables:
  BFL_API_KEY        — Black Forest Labs API key
  ANTHROPIC_API_KEY  — Anthropic API key
"""

import os
import sys
import json
import time
import base64
import argparse
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BFL_API_KEY = os.environ.get("BFL_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
KONTEXT_ENDPOINT = "https://api.bfl.ai/v1/flux-kontext-pro"
KONTEXT_RESULT_ENDPOINT = "https://api.bfl.ai/v1/get_result"
MAX_ATTEMPTS = 3
POLL_INTERVAL = 3


CORRECT_LAYOUT = """
The Chubby Buttons remote has 6 large round rubber buttons arranged in
a 2-column x 3-row grid on an olive/army green matte plastic body,
plus a small power button and a small Bluetooth button.

Correct symbol layout when the remote is worn on the forearm
(reading from ELBOW end toward WRIST):

  Row 1 (nearest elbow):  Volume Down (horizontal line)  |  Power icon (small, top edge)
  Row 2 (middle):         Skip Back (double left triangles)  |  Play/Pause (right triangle with pause bars)
  Row 3 (nearest wrist):  Skip Forward (double right triangles)  |  Volume Up (plus sign)

The strap reads "Chubby Buttons" in a woven pattern.
All button symbols are subtle embossed lines in yellow-green on olive buttons.
"""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def encode_image_bytes(data):
    return base64.b64encode(data).decode()


def get_media_type(path):
    ext = Path(path).suffix.lower().lstrip(".")
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"}.get(ext, "image/png")


def assess_symbols(image_b64, media_type="image/png"):
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64",
                                   "media_type": media_type,
                                   "data": image_b64},
                    },
                    {
                        "type": "text",
                        "text": f"""You are a QA inspector for Chubby Buttons product marketing images.

Examine the wearable Bluetooth remote in this image. Here is the CORRECT layout:

{CORRECT_LAYOUT}

Assess:
1. Is a remote visible in the image?
2. Are the button symbols correct and in the right order?
3. What specific issues do you see (wrong symbols, flipped order, garbled icons, missing buttons)?
4. What is the remote's orientation on the arm (horizontal/vertical/angled)?

Then write a concise FLUX Kontext editing instruction to fix the symbols.
The instruction should be specific about which symbols to change and where.

Respond ONLY in this JSON format, no other text:
{{
  "remote_found": true/false,
  "symbols_correct": true/false,
  "orientation": "horizontal|vertical|angled",
  "issues": ["issue 1", "issue 2"],
  "correction_instruction": "Change the button symbols on the wearable remote so that..."
}}"""
                    },
                ],
            }],
        },
    )

    print(f"      Claude API status: {response.status_code}")
    data = response.json()
    if "content" not in data:
        raise RuntimeError(f"Claude API error: {data}")
    text = data["content"][0]["text"]
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def call_kontext(image_b64, instruction):
    response = requests.post(
        KONTEXT_ENDPOINT,
        headers={
            "accept": "application/json",
            "x-key": BFL_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "input_image": image_b64,
            "prompt": instruction,
            "output_format": "png",
        },
    )
    print(f"      Kontext API status: {response.status_code}")
    result = response.json()
    task_id = result.get("id")
    polling_url = result.get("polling_url")

    if not task_id:
        raise RuntimeError(f"Kontext submission failed: {result}")

    print(f"      Kontext task ID: {task_id}")

    for i in range(60):
        time.sleep(POLL_INTERVAL)
        poll = requests.get(
            polling_url or f"{KONTEXT_RESULT_ENDPOINT}?id={task_id}",
            headers={
                "accept": "application/json",
                "x-key": BFL_API_KEY,
            },
        )
        poll_data = poll.json()
        status = poll_data.get("status")
        print(f"      Poll {i+1}: {status}")

        if status == "Ready":
            image_url = poll_data["result"]["sample"]
            img_response = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"})
            return encode_image_bytes(img_response.content)

        if status in ("Error", "Failed"):
            raise RuntimeError(f"Kontext generation failed: {poll_data}")

    raise TimeoutError("Kontext generation timed out after 3 minutes")


def qa_check(image_b64, media_type="image/png"):
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64",
                                   "media_type": media_type,
                                   "data": image_b64},
                    },
                    {
                        "type": "text",
                        "text": f"""You are a QA inspector for Chubby Buttons marketing images.

Check the wearable remote's button symbols against this correct layout:

{CORRECT_LAYOUT}

Score the symbol accuracy from 1-10:
- 10: Perfect match to correct layout
- 7-9: Mostly correct, minor issues
- 4-6: Some symbols right, some wrong
- 1-3: Mostly wrong or garbled

A score of 7+ passes QA.

Respond ONLY in JSON:
{{
  "pass": true/false,
  "score": <1-10>,
  "remaining_issues": ["issue 1", "issue 2"]
}}"""
                    },
                ],
            }],
        },
    )

    print(f"      Claude QA status: {response.status_code}")
    data = response.json()
    if "content" not in data:
        raise RuntimeError(f"Claude QA error: {data}")
    text = data["content"][0]["text"]
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def fix_remote_symbols(image_path, output_path=None, reference_path=None):
    if not output_path:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_fixed.png")

    media_type = get_media_type(image_path)
    image_b64 = encode_image(image_path)

    print("[1/3] Assessing symbols with Claude Vision...")
    print(f"      ANTHROPIC_API_KEY set: {bool(ANTHROPIC_API_KEY)} (len={len(ANTHROPIC_API_KEY)})")
    print(f"      BFL_API_KEY set: {bool(BFL_API_KEY)} (len={len(BFL_API_KEY)})")

    assessment = assess_symbols(image_b64, media_type)
    print(f"      Remote found: {assessment['remote_found']}")
    print(f"      Symbols correct: {assessment['symbols_correct']}")

    if not assessment["remote_found"]:
        return {"success": False, "reason": "no_remote", "attempts": 0}

    if assessment["symbols_correct"]:
        return {"success": True, "reason": "already_correct", "attempts": 0}

    print(f"      Issues: {assessment['issues']}")

    current_b64 = image_b64
    best_score = 0
    best_b64 = None
    qa_result = None
    attempt = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n[2/3] Attempt {attempt}/{MAX_ATTEMPTS}: Fixing with FLUX Kontext Pro...")

        instruction = assessment["correction_instruction"]
        if attempt > 1 and qa_result and qa_result.get("remaining_issues"):
            instruction += (
                f" IMPORTANT: Previous attempt still had these issues: "
                f"{qa_result['remaining_issues']}. Fix these specifically."
            )

        print(f"      Instruction: {instruction[:120]}...")

        try:
            fixed_b64 = call_kontext(current_b64, instruction)
        except Exception as e:
            print(f"      Kontext error: {e}")
            continue

        print(f"[3/3] QA checking attempt {attempt}...")
        qa_result = qa_check(fixed_b64)
        print(f"      Score: {qa_result['score']}/10, Pass: {qa_result['pass']}")

        if qa_result["score"] > best_score:
            best_score = qa_result["score"]
            best_b64 = fixed_b64

        if qa_result["pass"]:
            print(f"\n      QA PASSED on attempt {attempt}!")
            break

        if qa_result.get("remaining_issues"):
            print(f"      Remaining issues: {qa_result['remaining_issues']}")

        current_b64 = best_b64

    if best_b64:
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(best_b64))
        print(f"\n      Saved to: {output_path}")

    return {
        "success": best_score >= 7,
        "attempts": attempt,
        "final_score": best_score,
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Flask server
# ---------------------------------------------------------------------------
from flask import Flask, request as flask_request, jsonify, send_file
import tempfile

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "cb-symbol-fixer",
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "bfl_key_set": bool(BFL_API_KEY),
    })


@app.route("/fix", methods=["POST"])
def fix():
    data = flask_request.json
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "image_url required"}), 400

    try:
        img_data = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"}).content
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_data)
            tmp_path = tmp.name

        output_path = tmp_path.replace(".png", "_fixed.png")
        result = fix_remote_symbols(tmp_path, output_path)
        os.unlink(tmp_path)

        if result["success"] and os.path.exists(output_path):
            return send_file(output_path, mimetype="image/png")
        else:
            return jsonify(result), 422
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix Chubby Buttons remote symbols")
    parser.add_argument("image", nargs="?", help="Path to image to fix")
    parser.add_argument("--reference", help="Path to reference product photo")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--serve", action="store_true", help="Run as Flask server")
    args = parser.parse_args()

    if args.serve:
        app.run(host="0.0.0.0", port=5050)
    elif args.image:
        result = fix_remote_symbols(args.image, args.output, args.reference)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
