import json
import time
import google.generativeai as genai

def parse_gemini_response(response_text: str) -> dict:
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

def validate_gap_report(report: dict) -> bool:
    if "gaps" not in report or not isinstance(report["gaps"], list):
        return False
    if len(report["gaps"]) == 0:
        return False
    valid_types = {"Population Gap", "Methodology Gap", "Contextual Gap",
                   "Temporal Gap", "Contradiction Gap", "Application Gap"}
    valid_priorities = {"High", "Medium", "Exploratory"}
    for gap in report["gaps"]:
        required = ["type", "title", "description", "evidence_themes", "priority", "suggested_direction"]
        if not all(k in gap for k in required):
            return False
        if gap["type"] not in valid_types:
            return False
        if gap["priority"] not in valid_priorities:
            return False
        if not isinstance(gap["evidence_themes"], list):
            return False
    return True

def call_gemini_with_retry(prompt: str, max_retries=3) -> str:
    """Call Gemini and handle 429 errors."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota exceeded" in error_str:
                try:
                    delay = float(error_str.split("retry in ")[1].split("s.")[0])
                except:
                    delay = 60
                print(f"Rate limit on gap analysis. Waiting {delay}s...")
                time.sleep(delay + 1)
            else:
                print(f"API error: {e}. Retrying in 5s...")
                time.sleep(5)
    return ""

def perform_gap_analysis(summaries: list, user_topic: str, n_papers: int) -> dict:
    themes_text = ""
    for i, s in enumerate(summaries):
        sparse_label = "sparse - understudied" if s.get("sparse", False) else "well-covered"
        themes_text += f"Theme {i} ({s['percentage']}% of literature, {sparse_label}):\n{s['summary']}\n\n"

    prompt = f"""You are an expert academic research analyst with deep knowledge of systematic literature review methodology.

A researcher has uploaded {n_papers} academic papers on the topic: "{user_topic}"

Below are the major research themes found across this literature, derived from semantic analysis:

RESEARCH THEMES COVERED:
{themes_text}

TASK:
Identify genuine research gaps. A gap is NOT something mentioned as future work by authors. 
It is something the collective literature has STRUCTURALLY failed to address.

Classify each gap as one of:
- Population Gap: studied in one group, not another
- Methodology Gap: only one approach type used across all papers  
- Contextual Gap: works in one setting, untested elsewhere
- Temporal Gap: research predates major developments
- Contradiction Gap: papers reach opposite conclusions with no reconciliation
- Application Gap: theory well-developed, real-world deployment missing

For each gap provide:
1. Type (from the list above)
2. Title (one clear sentence naming the gap)
3. Description (2-3 sentences: what is missing and why it matters)
4. Evidence (which theme numbers support this gap existing)
5. Priority: High / Medium / Exploratory
6. Suggested direction (one concrete next research step)

Return ONLY valid JSON. No explanation before or after. No markdown code fences.

{{
  "domain": "inferred domain from themes",
  "total_papers_analysed": {n_papers},
  "themes_found": {len(summaries)},
  "gaps": [
    {{
      "type": "...",
      "title": "...",
      "description": "...",
      "evidence_themes": [0, 2],
      "priority": "High",
      "suggested_direction": "..."
    }}
  ]
}}"""

    response_text = call_gemini_with_retry(prompt)
    if not response_text:
        return {"error": "Failed to get response after retries."}

    for attempt in range(2):  # two attempts at parsing/validation
        try:
            report = parse_gemini_response(response_text)
        except Exception:
            # retry with instruction
            prompt2 = prompt + "\n\nReturn only valid JSON matching the schema exactly."
            response_text = call_gemini_with_retry(prompt2)
            if not response_text:
                return {"error": "Failed after parse error."}
            continue  # go back to parse
        if validate_gap_report(report):
            return report
        else:
            # Validation failed, retry
            prompt2 = prompt + "\n\nReturn only valid JSON matching the schema exactly."
            response_text = call_gemini_with_retry(prompt2)
            if not response_text:
                return {"error": "Validation failed and retry empty."}
    return {"error": "Validation failed after multiple attempts."}
