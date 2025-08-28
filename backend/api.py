import os
import re
import math
import time
import atexit
import signal
import logging
import threading
from collections import defaultdict
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from flask_cors import CORS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# Config
# =========================
DEFAULT_START_URL = "https://en.wikipedia.org/wiki/Python_(programming_language)"
STARTING_URL = os.getenv("STARTING_URL", DEFAULT_START_URL)
MAX_PAGES_TO_CRAWL = int(os.getenv("MAX_PAGES_TO_CRAWL", "20"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.8"))

USER_AGENT = os.getenv(
    "USER_AGENT",
    "SimpleSearchBot/1.1 (+https://example.com/resume; for demo; contact=you@example.com)"
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STOP_WORDS = {
    'a', 'an', 'the', 'and', 'in', 'is', 'it', 'of', 'for', 'on', 'with', 'to',
    'was', 'as', 'by', 'that', 'this', 'at', 'from', 'but', 'or', 'be', 'are'
}

# =========================
# Utilities
# =========================
def make_retrying_session() -> requests.Session:
    """Create a requests session with retries and polite headers."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session

# =========================
# Indexer
# =========================
class Indexer:
    """Thread-safe inverted index with TF-IDF scoring."""
    def __init__(self):
        self._lock = threading.RLock()
        self.inverted_index: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.documents: dict[str, str] = {}
        self.idf_scores: dict[str, float] = {}

    def _preprocess_text(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return [t for t in tokens if t not in STOP_WORDS]

    def rebuild(self, crawled_data: dict[str, str]) -> None:
        """Rebuild index from scratch."""
        with self._lock:
            logger.info("Index: rebuilding from %d documents...", len(crawled_data))
            self.inverted_index.clear()
            self.documents.clear()
            self.idf_scores.clear()

            self.documents = crawled_data

            # TF
            for url, text in self.documents.items():
                for token in self._preprocess_text(text):
                    self.inverted_index[token][url] += 1

            # IDF
            n_docs = len(self.documents)
            if n_docs == 0:
                logger.warning("Index: no documents to index.")
                return

            for token, doc_freqs in self.inverted_index.items():
                num_docs_with_token = len(doc_freqs)
                # +1 in denominator to avoid div-by-zero if something odd happens
                self.idf_scores[token] = math.log(n_docs / (1 + num_docs_with_token))

            logger.info("Index: done. %d unique terms.", len(self.inverted_index))

    def _snippet(self, text: str, query_tokens: list[str], length: int = 250) -> str:
        lower = text.lower()
        first = -1
        for tok in query_tokens:
            idx = lower.find(tok)
            if idx != -1:
                first = idx
                break

        if first == -1:
            return (text[:length] + "...") if len(text) > length else text

        start = max(0, first - length // 2)
        end = min(len(text), start + length)
        snippet = text[start:end]
        # highlight
        for tok in query_tokens:
            snippet = re.sub(f"({re.escape(tok)})", r"<strong>\1</strong>", snippet, flags=re.IGNORECASE)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{snippet}{suffix}"

    def search(self, query: str) -> list[dict]:
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return []

        with self._lock:
            # start with first token postings
            first = query_tokens[0]
            if first not in self.inverted_index:
                return []

            matching = set(self.inverted_index[first].keys())
            for tok in query_tokens[1:]:
                matching &= set(self.inverted_index.get(tok, {}).keys())

            if not matching:
                return []

            scores: dict[str, float] = {}
            for url in matching:
                s = 0.0
                for tok in query_tokens:
                    tf = self.inverted_index.get(tok, {}).get(url, 0)
                    idf = self.idf_scores.get(tok, 0.0)
                    s += tf * idf
                scores[url] = s

            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            results = []
            for url, score in ranked:
                results.append({
                    "id": url,
                    "score": round(score, 3),
                    "snippet": self._snippet(self.documents.get(url, ""), query_tokens)
                })
            return results

# =========================
# Crawler
# =========================
class WebCrawler:
    """Simple same-domain crawler with politeness and stop flag."""
    def __init__(self, start_url: str, max_pages: int, stop_event: threading.Event):
        self.start_url = start_url
        self.max_pages = max_pages
        self.stop_event = stop_event

        self.base_netloc = urlparse(start_url).netloc
        self.visited: set[str] = set()
        self.queue: list[str] = [start_url]
        self.data: dict[str, str] = {}
        self.session = make_retrying_session()

    def _same_domain(self, url: str) -> bool:
        p = urlparse(url)
        return p.scheme in ("http", "https") and p.netloc == self.base_netloc

    def crawl(self) -> dict[str, str]:
        logger.info("Crawl: starting at %s (max=%d)", self.start_url, self.max_pages)
        while self.queue and len(self.data) < self.max_pages and not self.stop_event.is_set():
            url = self.queue.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)

            logger.info("Crawl: (%d/%d) %s", len(self.data) + 1, self.max_pages, url)
            try:
                resp = self.session.get(url, timeout=12)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning("Crawl: failed %s: %s", url, e)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            text = " ".join(s.strip() for s in soup.stripped_strings)
            self.data[url] = text

            # discover links
            for a in soup.find_all("a", href=True):
                href = a["href"]
                abs_url = urljoin(url, href).split("#")[0]
                if self._same_domain(abs_url) and abs_url not in self.visited:
                    self.queue.append(abs_url)

            # polite delay
            time.sleep(REQUEST_DELAY)

        logger.info("Crawl: finished with %d pages.", len(self.data))
        return self.data

# =========================
# Search Service (background worker)
# =========================
class SearchService:
    """Owns the indexer, manages background indexing and readiness."""
    def __init__(self, start_url: str, max_pages: int):
        self.indexer = Indexer()
        self.ready_event = threading.Event()
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_once = False
        self.start_url = start_url
        self.max_pages = max_pages
        self._guard_lock = threading.Lock()

    def start_in_background(self):
        """Start background indexing only once per process."""
        with self._guard_lock:
            if self._started_once:
                logger.info("SearchService: background already started; skipping.")
                return
            self._started_once = True

            def _worker():
                logger.info("SearchService: worker started.")
                try:
                    crawler = WebCrawler(self.start_url, self.max_pages, self.stop_event)
                    data = crawler.crawl()
                    if data:
                        self.indexer.rebuild(data)
                        self.ready_event.set()
                        logger.info("SearchService: ready ✅")
                    else:
                        logger.error("SearchService: no data crawled; staying not-ready.")
                except Exception as e:
                    logger.exception("SearchService: fatal worker error: %s", e)
                finally:
                    logger.info("SearchService: worker finished.")

            self._thread = threading.Thread(target=_worker, name="indexer-worker")
            # Non-daemon to avoid stderr lock errors on shutdown
            self._thread.start()

    def stop(self, join_timeout: float = 8.0):
        """Signal worker to stop and join."""
        self.stop_event.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=join_timeout)

    def is_ready(self) -> bool:
        return self.ready_event.is_set()

    def search(self, q: str) -> list[dict]:
        return self.indexer.search(q)

# =========================
# Flask App Factory
# =========================
def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    service = SearchService(STARTING_URL, MAX_PAGES_TO_CRAWL)

    # Start background indexing exactly once per worker
    service.start_in_background()

    @app.route("/", methods=["GET"])
    def home():
        status = "ready" if service.is_ready() else "indexing"
        return jsonify({
            "message": "Backend is running",
            "status": status,
            "start_url": STARTING_URL,
            "max_pages": MAX_PAGES_TO_CRAWL
        }), 200

    @app.route("/search", methods=["GET"])
    def search_api():
        if not service.is_ready():
            return jsonify({"error": "Index not ready yet. Try again in a moment."}), 503

        query = request.args.get("q", "").strip()
        if not query:
            return jsonify({"error": "Query parameter 'q' is required."}), 400

        logging.info("Search: '%s'", query)
        results = service.search(query)
        return jsonify({
            "query": query,
            "count": len(results),
            "results": results
        }), 200

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify({
            "ready": service.is_ready(),
            "start_url": STARTING_URL,
            "max_pages": MAX_PAGES_TO_CRAWL
        }), 200

    # Optional: manual reindex (protect with a token in real deployments)
    @app.route("/admin/reindex", methods=["POST"])
    def admin_reindex():
        # Very simple protection (improve for production)
        token = request.headers.get("X-Admin-Token", "")
        expected = os.getenv("ADMIN_TOKEN", "")
        if not expected or token != expected:
            return jsonify({"error": "Unauthorized"}), 401

        # Restart background worker
        service.stop()
        # Reset readiness to force 503 until new index is built
        service.ready_event.clear()
        service.stop_event.clear()
        service._started_once = False
        service.start_in_background()
        return jsonify({"message": "Reindex started"}), 202

    # Graceful shutdown (Gunicorn/Flask)
    def _graceful_shutdown(*_):
        logger.info("Signal received: shutting down gracefully…")
        service.stop()

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    atexit.register(service.stop)

    # Attach service on app for debugging/tests if needed
    app.search_service = service  # type: ignore[attr-defined]
    return app

# Gunicorn entrypoint: `gunicorn -w 2 -b 0.0.0.0:8000 api:app`
app = create_app()

if __name__ == "__main__":
    # Flask dev run (single process). Avoid the reloader spawning duplicate workers.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False, use_reloader=False)
