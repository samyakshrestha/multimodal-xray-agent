"""

This module provides utility functions for processing and generating question-answer
pairs from chest X-ray impression texts, particularly from the IU-Xray dataset.
It includes functions for extracting impression sections, detecting abnormal findings,
normalizing text, handling redacted tokens, paraphrasing impressions using GPT-4o,
and dumping data to disk in JSONL format.

Key components:
- ABNORMAL_KEYWORDS: Set of keywords indicating abnormal radiologic findings.
- REDACTION_TOKEN / REDACTION_REPLACEMENT: Constants for handling redacted text.
- extract_impression: Extracts the impression section from a radiology report.
- has_abnormal_keyword: Checks if impression contains abnormal findings.
- normalize_start: Normalizes the start of impression text for comparison.
- replace_redactions: Replaces redacted tokens with a human-readable placeholder.
- gpt_paraphrase: Uses OpenAI GPT-4o to paraphrase impressions into concise summaries.
- dump: Writes a list of dictionaries to disk in JSONL format.

Note: Requires OpenAI API credentials to be set in the environment.
qa_utils.py — Helper functions for QA pair generation from IU-Xray impressions.
This module is used in 06_generate_qa_pairs.ipynb.
"""

import json
import re
from openai import OpenAI

# Constants used across functions
ABNORMAL_KEYWORDS = set([
    "atelectasis", "consolidation", "effusion", "opacity", "nodule", "mass",
    "infiltrate", "pneumothorax", "edema", "cardiomegaly", "fracture",
    "emphysema", "interstitial", "pleural", "pneumonia", "hernia"
])
REDACTION_TOKEN = "XXXX"
REDACTION_REPLACEMENT = "[REDACTED]"

# Initialize OpenAI client from environment
client = OpenAI()

def extract_impression(text):
    """Extracts the impression section from a report, if labeled."""
    if "IMPRESSION:" in text:
        return text.split("IMPRESSION:")[-1].strip(" .:\n").strip()
    return text.strip(" .:\n").strip()


def has_abnormal_keyword(text):
    """Returns True if impression contains signs of abnormality."""
    words = text.lower().split()
    return any(kw in words for kw in ABNORMAL_KEYWORDS)


def normalize_start(text):
    """Normalizes impression start for canonical tiering comparisons."""
    return re.sub(r"^[\W\d]*", "", text.lower()).strip()


def replace_redactions(text):
    """Replaces redacted tokens (XXXX) with human-readable placeholder."""
    return text.replace(REDACTION_TOKEN, REDACTION_REPLACEMENT)


def gpt_paraphrase(impression):
    """
    Uses GPT-4o to paraphrase the impression into a short radiologic summary.
    Ensures findings are preserved and avoids hallucinations.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical summarizer. Rephrase the following chest X-ray impression "
                "as a short radiologic summary, preserving all findings exactly. Do not invent new terms. "
                "Output should be under 25 tokens."
            ),
        },
        {"role": "user", "content": impression.strip()}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT error: {e}")
        return None


def dump(path, data):
    """Writes a list of dictionaries to disk in JSONL format."""
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in data)