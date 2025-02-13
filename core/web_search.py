# core/web_search.py

import requests
import trafilatura
from bs4 import BeautifulSoup
import wikipedia
from dateutil import parser as date_parser
from datetime import datetime
import ollama

import sys_msgs

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
        if i > 9:
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
        summary_text = wikipedia.summary(page_title, sentences=5)
        return f"**Wikipedia Page**: {page_title}\n\n{summary_text}"
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
        model="llama3.1",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt_text}
        ]
    )
    return resp["message"]["content"].strip()

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

# At the top of core/web_search.py (after your imports)
from core import db_utils
def get_recent_user_queries(num_messages: int = 2) -> str:
    """
    Extracts the last `num_messages` user queries from the chat history (ignoring assistant messages)
    and returns them concatenated into a single string.
    """
    from core import db_utils
    # Filter only user messages
    user_msgs = [msg["content"] for msg in db_utils.chat_history if msg["role"] == "user"]
    recent_queries = user_msgs[-num_messages:]  # take last two queries
    # Concatenate with a space; you can adjust formatting if desired.
    return " ".join(recent_queries).strip()


def refine_search_query(user_input: str) -> str:
    """
    Creates a concise external search query using only the recent user queries.
    """
    recent_context = get_recent_user_queries(num_messages=2)
    # For a more concise query, you might simply return the latest user query.
    # Alternatively, if you want to combine them, you can do:
    refined = recent_context  # Using only user messages for brevity
    return refined

def generate_web_search_query(user_input: str) -> str:
    """
    Uses an LLM-based agent to generate a concise and effective search query
    based on recent conversation context and the current user query.
    """
    from core import db_utils  # to access global chat_history
    # Extract the last 2 user messages from chat history.
    user_msgs = [msg["content"] for msg in db_utils.chat_history if msg["role"] == "user"]
    recent_context = " ".join(user_msgs[-2:]).strip() if user_msgs else ""
    
    # Construct the prompt for the query generator agent.
    prompt = f"Conversation context: {recent_context}\nCurrent query: {user_input}\n\nGenerate a concise search query for retrieving up-to-date information:"
    
    resp = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": sys_msgs.web_query_generator_msg},
            {"role": "user", "content": prompt}
        ]
    )
    query = resp["message"]["content"].strip()
    # Remove any leading and trailing double quotes.
    if query.startswith('"') and query.endswith('"'):
        query = query[1:-1].strip()
    print(f"[generate_web_search_query] Generated query: '{query}'")
    return query


def web_search_flow(user_input: str) -> str:
    # Use the dedicated agent to generate a refined search query.
    refined_query = generate_web_search_query(user_input)
    print(f"[web_search_flow] Final search query: '{refined_query}'")
    
    # Ask the source-deciding agent with the refined query.
    resp = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": sys_msgs.source_decider_msg},
            {"role": "user", "content": refined_query}
        ]
    )
    source = resp["message"]["content"].strip().lower()
    print(f"[web_search_flow] source-decider agent says: '{source}'")
    
    # Depending on the source-decider output, use Wikipedia or news search.
    if source == "wiki":
        result = wikipedia_flow(refined_query)
    else:
        result = gather_news_articles(refined_query)
    return result
