# utils/scraper.py
from ddgs import DDGS

def ddg_search_internships(role, location, top_k=5):
    """
    Scrape internships using DuckDuckGo search (ddgs package)
    Returns a list of dictionaries with title, link, snippet
    """
    query = f'"{role} internship in {location}"'
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=top_k):
                results.append({
                    "title": r.get("title", "No Title"),
                    "link": r.get("href", "#"),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        print("⚠️ DuckDuckGo scraping failed:", e)
        return []

    return results
