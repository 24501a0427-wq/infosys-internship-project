"""
MILESTONE 3 - Gemini LLM Connection
-----------------------------------
Send prompts to Gemini with retry handling.
"""

import os
import time

from dotenv import load_dotenv
from google import genai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def has_key():
    return bool(os.getenv("GEMINI_API_KEY"))


def ask_llm(prompt):
    """Send a prompt to Gemini with retries."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or not prompt.strip():
        return None

    client = genai.Client(api_key=api_key)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )

            return response.text.strip() if response.text else None

        except Exception as e:
            error = str(e)

            if "503" in error or "429" in error:
                wait = 2 ** attempt
                print(f"Gemini temporarily unavailable. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Gemini error: {e}")
                return None

    print("Gemini unavailable after 3 attempts.")
    return None


if __name__ == "__main__":
    result = ask_llm("Explain PPE in one sentence.")

    print(result or "Gemini unavailable. Check GEMINI_API_KEY and GEMINI_MODEL.")