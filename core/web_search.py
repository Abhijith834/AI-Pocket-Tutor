import requests
import trafilatura
from bs4 import BeautifulSoup
import wikipedia
from dateutil import parser as date_parser
from datetime import datetime
import ollama
import sys_msgs
import core.config as config
from core.db_utils import remove_think_clauses
from core import db_utils

def extract_publication_date(html: str):
    soup = BeautifulSoup(html, "html.parser")
    def parse_naive(ds: str):
        dt = date_parser.parse(ds)
        return dt.replace(tzinfo=None)
    for tag in soup.find_all("meta"):
        if tag.get("property") == "article:published_time":
            ds = tag.get("content")
            try:
                return parse_naive(ds)
            except Exception:
                pass
        if tag.get("name") in ["pubdate", "publish-date", "publish_date"]:
            ds = tag.get("content")
            try:
                return parse_naive(ds)
            except Exception:
                pass
    ttag = soup.find("time", {"datetime": True})
    if ttag:
        ds = ttag["datetime"]
        try:
            return parse_naive(ds)
        except Exception:
            pass
    return None

def scrape_webpage(url: str):
    print(f"[Web] Scraping webpage: {url}")
    try:
        downloaded = trafilatura.fetch_url(url=url)
        extracted = trafilatura.extract(downloaded, include_formatting=True, include_links=True)
        if not extracted:
            print(f"[Web] No text extracted from {url}.")
            return None, None
        return extracted, downloaded
    except Exception as e:
        print(f"[Web] Failed to scrape {url}: {e}")
        return None, None

def duckduckgo_search(query: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.36"
        )
    }
    encoded = requests.utils.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    print(f"[Web] Performing DuckDuckGo search with query: '{query}'\nURL: {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Web] DuckDuckGo request failed: {e}")
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    containers = soup.find_all("div", class_="result__body")
    if not containers:
        containers = soup.find_all("div", class_="result")
    results = []
    for i, div in enumerate(containers):
        # Limit to 5 results
        if i > 4:
            break
        a_tag = div.find("a", class_="result__a")
        if not a_tag:
            continue
        link = a_tag.get("href", "")
        snippet_tag = div.find("a", class_="result__snippet")
        snippet = snippet_tag.text.strip() if snippet_tag else "No description available"
        date_tag = div.find("span", class_="result__date")
        date_text = date_tag.text.strip() if date_tag else ""
        if date_text:
            snippet += f" (Date: {date_text})"
        results.append({
            "id": i,
            "link": link,
            "search_description": snippet
        })
    print(f"[Web] Retrieved {len(results)} results from DuckDuckGo.")
    return results

def wikipedia_flow(query: str) -> str:
    print(f"[Web] Using Wikipedia for query: {query}")
    try:
        results = wikipedia.search(query)
        if not results:
            print("[Web] No Wikipedia pages found.")
            return ""
        page_title = results[0]
        try:
            page = wikipedia.page(page_title)
            page_url = page.url
        except Exception as e:
            print(f"[Web] Error retrieving page URL: {e}")
            page_url = "URL not available"
        summary_text = wikipedia.summary(page_title, sentences=5)
        summary_text = remove_think_clauses(summary_text)
        return (
            f"**Wikipedia Page**: [{page_title}]({page_url})\n\n"
            f"{summary_text}\n\n"
            f"Reference: This information was retrieved from Wikipedia using the query: '{query}'."
        )
    except Exception as e:
        print(f"[Web] Error during Wikipedia retrieval: {e}")
        return ""

def summarize_article_content(content: str, query: str, pub_date_str: str, link: str) -> str:
    sys_msg = (
        "You are an article summarization agent. Summarize the given article content, focusing on key details "
        "that are relevant to the user's query (such as events, outcomes, or newsworthy points)."
    )
    prompt_text = (
        f"Article Link: {link}\n"
        f"Publication Date: {pub_date_str}\n"
        f"User Query: {query}\n\n"
        f"Article Content:\n{content}\n\n"
        "Provide a concise summary focusing on the most important details relevant to the query."
    )
    resp = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt_text}
        ]
    )
    raw_summary = resp["message"]["content"].strip()
    clean_summary = remove_think_clauses(raw_summary)
    return clean_summary

def gather_news_articles(query: str) -> str:
    ddg_results = duckduckgo_search(query)
    if not ddg_results:
        print("[Web] No DuckDuckGo results found.")
        return ""
    scraped = []
    for r in ddg_results:
        text, raw_html = scrape_webpage(r["link"])
        if not text:
            continue
        pub_date = extract_publication_date(raw_html)
        r["page_text"] = text
        r["publication_date"] = pub_date
        scraped.append(r)
    if not scraped:
        print("[Web] No articles could be scraped.")
        return ""
    def sort_key(x):
        return x["publication_date"] if x["publication_date"] else datetime.min
    scraped.sort(key=sort_key, reverse=True)
    combined_summary = []
    for idx, article in enumerate(scraped):
        date_str = str(article["publication_date"]) if article["publication_date"] else "N/A"
        content_text = article["page_text"]
        if len(content_text) > 5000:
            content_text = content_text[:5000] + "..."
        summary = summarize_article_content(content_text, query, date_str, article["link"])
        combined_summary.append(
            f"Article {idx}:\nLink: {article['link']}\nPublication Date: {date_str}\nSummary:\n{summary}\n"
        )
    return "\n".join(combined_summary)

def refine_external_query(current_query: str, previous_query: str) -> str:
    """
    Uses an LLM-based agent to combine a previous external search query with the current query,
    producing a refined, concise search query that captures both contexts.
    """
    prompt = (
        f"Previous external search query: {previous_query}\n"
        f"Current user query: {current_query}\n\n"
        "Generate a concise combined search query that incorporates both contexts and is suitable for a search engine. "
        "Do not include any quotation marks or extra commentary."
    )
    resp = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": sys_msgs.web_query_generator_msg},
            {"role": "user", "content": prompt}
        ]
    )
    new_query = resp["message"]["content"].strip()
    new_query = remove_think_clauses(new_query)
    new_query = new_query.replace('"', '').replace("'", "").strip()
    print(f"[refine_external_query] Generated refined query: '{new_query}'")
    return new_query

def generate_web_search_query(user_input: str) -> str:
    """
    Uses an LLM-based agent to generate a concise search query based on the
    recent user queries and the current user query. Any quotation marks are removed.
    """
    user_msgs = [msg["content"] for msg in db_utils.chat_history if msg["role"] == "user"]
    recent_context = " ".join(user_msgs[-2:]).strip() if user_msgs else ""

    prompt = (
        f"Conversation context: {recent_context}\n"
        f"Current query: {user_input}\n\n"
        "Generate a concise search query suitable for a search engine. "
        "Do not include quotation marks or extra commentary."
    )
    resp = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": sys_msgs.web_query_generator_msg},
            {"role": "user", "content": prompt}
        ]
    )
    query = resp["message"]["content"].strip()
    query = remove_think_clauses(query)
    query = query.replace('"', '').replace("'", "").strip()
    print(f"[generate_web_search_query] Generated query: '{query}'")
    return query

last_external_context = ""

def web_search_flow(user_input: str) -> str:
    """
    Main entry point for an external web search. If a previous query is stored,
    we refine it with the current user query. Otherwise, we generate a fresh query.
    Then we pass this query to the source-decider agent and subsequently to
    Wikipedia or DuckDuckGo news search.
    """
    global last_external_context

    if last_external_context:
        refined_query = refine_external_query(user_input, last_external_context)
    else:
        refined_query = generate_web_search_query(user_input)

    print(f"[web_search_flow] Final search query: '{refined_query}'")

    resp = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": sys_msgs.source_decider_msg},
            {"role": "user", "content": refined_query}
        ]
    )
    raw_source = resp["message"]["content"].strip()
    raw_source = remove_think_clauses(raw_source)
    source = raw_source.lower()
    print(f"[web_search_flow] source-decider agent says: '{source}'")

    if source == "wiki":
        result = wikipedia_flow(refined_query)
    else:
        result = gather_news_articles(refined_query)

    if result.strip():
        last_external_context = refined_query

    return result
