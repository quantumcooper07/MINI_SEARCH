import logging
import math
import re
import time
from collections import defaultdict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Crawler Configuration ---
STARTING_URL = "https://en.wikipedia.org/wiki/Python_(programming_language)"
MAX_PAGES_TO_CRAWL = 20
REQUEST_DELAY = 1
HEADERS = {'User-Agent': 'SimpleSearchBot/1.0 (Project for resume)'}

# --- Stop Words ---
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'in', 'is', 'it', 'of', 'for', 'on', 'with', 'to',
    'was', 'as', 'by', 'that', 'this', 'at', 'from', 'but', 'or', 'be', 'are'
}


class Indexer:
    """Manages the indexing of web content and searching."""

    def __init__(self):
        """Initializes the Indexer with empty data structures."""
        self.inverted_index: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.documents: dict[str, str] = {}
        self.idf_scores: dict[str, float] = {}

    def _preprocess_text(self, text: str) -> list[str]:
        """
        Cleans and tokenizes text by converting to lowercase, finding all word
        sequences, and removing stop words.

        Args:
            text: The raw text content from a web page.

        Returns:
            A list of cleaned tokens.
        """
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return [token for token in tokens if token not in STOP_WORDS]

    def create_index(self, crawled_data: dict[str, str]) -> None:
        """
        Builds an inverted index with TF-IDF scores from crawled data.

        Args:
            crawled_data: A dictionary mapping URLs to their text content.
        """
        logging.info("Starting the indexing process...")
        self.documents = crawled_data
        
        # Step 1: Calculate Term Frequency (TF)
        for url, text in self.documents.items():
            tokens = self._preprocess_text(text)
            for token in tokens:
                self.inverted_index[token][url] += 1
        
        # Step 2: Calculate Inverse Document Frequency (IDF)
        num_documents = len(self.documents)
        if num_documents == 0:
            logging.warning("No documents to index.")
            return

        for token, doc_freqs in self.inverted_index.items():
            num_docs_with_token = len(doc_freqs)
            # Add 1 to the denominator to avoid division by zero for words in all docs
            self.idf_scores[token] = math.log(num_documents / (1 + num_docs_with_token))

        logging.info(f"Indexing complete. Indexed {len(self.inverted_index)} unique words.")

    def _generate_snippet(self, text: str, query_tokens: list[str], length: int = 250) -> str:
        """
        Generates a snippet of text around the first found query term.

        Args:
            text: The full text of the document.
            query_tokens: The list of processed query tokens.
            length: The desired length of the snippet.

        Returns:
            A formatted string snippet with query terms highlighted.
        """
        text_lower = text.lower()
        first_match_index = -1
        
        for token in query_tokens:
            try:
                first_match_index = text_lower.index(token)
                break
            except ValueError:
                continue

        if first_match_index == -1:
            return text[:length] + "..." if len(text) > length else text

        start = max(0, first_match_index - length // 2)
        end = start + length
        snippet = text[start:end]

        for token in query_tokens:
            snippet = re.sub(f'({re.escape(token)})', r'<strong>\1</strong>', snippet, flags=re.IGNORECASE)
        
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{snippet}{suffix}"

    def search(self, query: str) -> list[dict]:
        """
        Performs a search on the index using the TF-IDF algorithm.

        Args:
            query: The user's search query.

        Returns:
            A list of result dictionaries, sorted by relevance score.
        """
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return []
        
        # Find documents that contain ALL words in the query
        try:
            matching_urls = set(self.inverted_index[query_tokens[0]].keys())
        except KeyError:
            return []
        
        for token in query_tokens[1:]:
            matching_urls.intersection_update(self.inverted_index.get(token, {}).keys())

        if not matching_urls:
            return []

        # Calculate TF-IDF score for each matching document
        url_scores = defaultdict(float)
        for url in matching_urls:
            score = 0.0
            for token in query_tokens:
                tf = self.inverted_index[token].get(url, 0)
                idf = self.idf_scores.get(token, 0)
                score += tf * idf
            url_scores[url] = score
            
        sorted_results = sorted(url_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Format results with snippets for the API response
        final_results = []
        for url, score in sorted_results:
            final_results.append({
                "id": url,
                "score": round(score, 2),
                "snippet": self._generate_snippet(self.documents[url], query_tokens)
            })
        return final_results


class WebCrawler:
    """A simple web crawler to fetch content from a single domain."""

    def __init__(self, start_url: str, max_pages: int):
        """
        Initializes the WebCrawler.

        Args:
            start_url: The URL to begin crawling from.
            max_pages: The maximum number of pages to crawl.
        """
        self.start_url = start_url
        self.max_pages = max_pages
        self.visited_urls: set[str] = set()
        self.url_queue: list[str] = [start_url]
        self.crawled_data: dict[str, str] = {}

    def _is_valid_url(self, url: str) -> bool:
        """Checks if a URL is valid and within the same domain."""
        parsed_start_url = urlparse(self.start_url)
        parsed_url = urlparse(url)
        return (parsed_url.scheme in ['http', 'https'] and
                parsed_url.netloc == parsed_start_url.netloc)

    def crawl(self) -> dict[str, str]:
        """
        Starts the crawling process.

        Returns:
            A dictionary mapping crawled URLs to their text content.
        """
        logging.info("Starting web crawl...")
        while self.url_queue and len(self.crawled_data) < self.max_pages:
            current_url = self.url_queue.pop(0)
            if current_url in self.visited_urls:
                continue
            
            logging.info(f"Crawling: {current_url}")
            try:
                response = requests.get(current_url, timeout=5, headers=HEADERS)
                response.raise_for_status()

                self.visited_urls.add(current_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                page_text = ' '.join(soup.stripped_strings)
                self.crawled_data[current_url] = page_text

                for link in soup.find_all('a', href=True):
                    absolute_url = urljoin(current_url, link['href']).split('#')[0]
                    if self._is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                        self.url_queue.append(absolute_url)
                
                time.sleep(REQUEST_DELAY)

            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to crawl {current_url}: {e}")

        logging.info(f"Crawl finished. Found {len(self.crawled_data)} pages.")
        return self.crawled_data


# --- Global Search Engine Instance ---
search_indexer = Indexer()


# --- API Endpoints ---
@app.route('/search', methods=['GET'])
def search_api():
    """The main search endpoint."""
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "Query parameter 'q' is required."}), 400
    
    logging.info(f"Received search query: '{query}'")
    results = search_indexer.search(query)
    
    return jsonify({
        "query": query,
        "count": len(results),
        "results": results
    })


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Initializing search engine...")
    crawler = WebCrawler(start_url=STARTING_URL, max_pages=MAX_PAGES_TO_CRAWL)
    crawled_data = crawler.crawl()

    if crawled_data:
        search_indexer.create_index(crawled_data)
        logging.info("Search engine is ready and running.")
    else:
        logging.error("Could not initialize search engine: No data was crawled.")

    # Use a production-ready WSGI server in a real deployment
    app.run(debug=False, port=5000)
