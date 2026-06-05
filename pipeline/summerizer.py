# pipeline/summariser.py
import time
import google.generativeai as genai
import os

def configure_gemini():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def summarise_cluster(rep_texts: list, cluster_id: int, max_retries=3) -> dict:
    """
    Stage 5: Summarise a single cluster using Gemini 2.5 Flash.
    Retries on rate limit errors with exponential backoff.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    excerpts = "\n\n".join(rep_texts)
    prompt = f"""You are an academic research analyst.

Below are text excerpts from multiple research papers that share a common theme.

Summarise in exactly 3-4 sentences:
1. What research topic or problem this cluster addresses
2. What methods or approaches are predominantly used
3. What findings or conclusions are commonly reached
4. Any limitations or future work mentioned

Be specific and factual. Only use information present in the excerpts.
Do not invent details.

Excerpts:
{excerpts}"""

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            # SUCCESS – sleep 12 seconds (safe for 5 RPM) then return
            time.sleep(12)
            summary = response.text.strip()
            return {
                "cluster_id": cluster_id,
                "summary": summary
            }
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota exceeded" in error_str:
                try:
                    delay = float(error_str.split("retry in ")[1].split("s.")[0])
                except:
                    delay = 60
                print(f"Rate limit hit. Waiting {delay}s before retry...")
                time.sleep(delay + 1)
            else:
                print(f"Non-rate error: {e}. Retrying in 5s...")
                time.sleep(5)
    # If all retries exhausted, return empty summary
    return {
        "cluster_id": cluster_id,
        "summary": "[Summary failed due to API errors]"
    }

def summarise_all_clusters(clusters: list) -> list:
    """
    Iterate over clusters, call Gemini for each with sleep.
    """
    summaries = []
    for cluster in clusters:
        rep_texts = cluster["representative_texts"]
        summary = summarise_cluster(rep_texts, cluster["cluster_id"])
        # Merge sparse info
        summary["sparse"] = cluster["sparse"]
        summary["chunk_count"] = cluster["chunk_count"]
        summary["percentage"] = cluster["percentage"]
        summaries.append(summary)
    return summaries
