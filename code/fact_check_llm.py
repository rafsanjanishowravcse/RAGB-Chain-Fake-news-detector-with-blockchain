import requests
import mimetypes
import os
import json
import re
import html
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from sarvamai import SarvamAI
from claim_storage import ClaimStorageManager
from sentence_transformers import SentenceTransformer
from blockchain_registry import (
    register_flagged_source,
    check_source_reputation,
    get_sources_by_publisher,
)

# Load environment variables from .env file
# Try loading from script directory first, then current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    print(f"✓ Loaded .env from: {env_path}")
else:
    load_dotenv()
    print(f"⚠️  .env not found at {env_path}, trying current directory")

grok_api_key = os.getenv('GROQ_API_KEY')
serp_dev_api_key = os.getenv('SERP_DEV_API_KEY')
sarvam_api_key = os.getenv('SARVAM_API_KEY')

# Strip whitespace from API keys (common issue)
if grok_api_key:
    grok_api_key = grok_api_key.strip()
if serp_dev_api_key:
    serp_dev_api_key = serp_dev_api_key.strip()
if sarvam_api_key:
    sarvam_api_key = sarvam_api_key.strip()

model_multi_query = os.getenv('model_multi_query') or 'llama-3.1-8b-instant'
model_summarizer = os.getenv('model_summarizer') or 'llama-3.1-8b-instant'
model_judge = os.getenv('model_judge') or 'llama-3.1-8b-instant'

# Check if required API keys are present
if not grok_api_key or grok_api_key == 'your_grok_api_key_here':
    print("⚠️  Warning: GROQ_API_KEY not set. Please update your .env file with a valid Groq API key.")
else:
    print(f"✓ GROQ_API_KEY loaded (length: {len(grok_api_key)})")
if not serp_dev_api_key or serp_dev_api_key == 'your_serper_dev_api_key_here':
    print("⚠️  Warning: SERP_DEV_API_KEY not set. Please update your .env file with a valid Serper Dev API key.")
else:
    print(f"✓ SERP_DEV_API_KEY loaded (length: {len(serp_dev_api_key)})")
if not sarvam_api_key or sarvam_api_key == 'your_sarvam_api_key_here':
    print("⚠️  Warning: SARVAM_API_KEY not set. Please update your .env file with a valid Sarvam AI API key.")
else:
    print(f"✓ SARVAM_API_KEY loaded")

empty_search = None

# Initialize shared SentenceTransformer model for reuse
shared_embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Initialize claim storage manager with shared model
claim_storage = ClaimStorageManager(embedding_model=shared_embedding_model)

BN_DIGIT_TABLE = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
REAL_KEYWORDS_EN = ["REAL", "TRUE", "ACCURATE", "AUTHENTIC"]
FAKE_KEYWORDS_EN = ["FAKE", "FALSE", "MISLEADING", "HOAX"]
UNSURE_KEYWORDS_EN = ["UNSURE", "UNCERTAIN", "UNKNOWN", "INCONCLUSIVE"]
REAL_KEYWORDS_BN = ["সত্য", "সঠিক", "যথার্থ", "বাস্তব", "আসল", "প্রামাণিক"]
FAKE_KEYWORDS_BN = ["মিথ্যা", "ভুয়া", "ভুল", "অসত্য", "গুজব", "প্রতারণা"]
UNSURE_KEYWORDS_BN = ["অনিশ্চিত", "সন্দেহ", "নিশ্চিত নয়", "অস্পষ্ট", "অপর্যাপ্ত"]
MISINFO_KEYWORDS_EN = ["MISINFORMATION", "DISINFORMATION", "MISINFO"]
MISINFO_KEYWORDS_BN = ["ভুল তথ্য", "মিথ্যা প্রচার", "মিসইনফর্মেশন"]
SCORE_KEYWORDS_EN = ["credibility", "confidence", "score"]
SCORE_KEYWORDS_BN = ["বিশ্বাস", "আত্মবিশ্বাস", "স্কোর"]
TITLE_PATTERN = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
REFUTE_KEYWORDS = [
    "debunk", "false", "hoax", "misleading", "refute", "not true", "ভুয়া", "মিথ্যা", "গুজব",
    "অসত্য", "ভুল দাবি", "আসল নয়"
]

def _normalize_digits(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.translate(BN_DIGIT_TABLE)

def _infer_classification(label_text: str) -> str:
    if not isinstance(label_text, str):
        return "UNSURE"
    text_upper = label_text.upper()
    text_bn = label_text

    if any(word in text_upper for word in MISINFO_KEYWORDS_EN) or any(word in text_bn for word in MISINFO_KEYWORDS_BN):
        return "MISINFORMATION"
    if any(word in text_upper for word in FAKE_KEYWORDS_EN) or any(word in text_bn for word in FAKE_KEYWORDS_BN):
        return "FAKE"
    if any(word in text_upper for word in REAL_KEYWORDS_EN) or any(word in text_bn for word in REAL_KEYWORDS_BN):
        return "REAL"
    if any(word in text_upper for word in UNSURE_KEYWORDS_EN) or any(word in text_bn for word in UNSURE_KEYWORDS_BN):
        return "UNSURE"
    return "UNSURE"

def _extract_score_from_text(text: str) -> int:
    if not isinstance(text, str):
        return 0
    normalized = _normalize_digits(text)
    patterns = [
        r'(100|\d{1,2})\s*/\s*100',
        r'(100|\d{1,2})\s*%',
        r'\b(100|\d{1,2})\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            try:
                score = int(match.group(1))
                return max(0, min(score, 100))
            except ValueError:
                continue
    return 0


def _build_onchain_metadata(source_url: Optional[str], publisher: Optional[str], classification: str) -> Dict[str, Any]:
    """Check reputation and (for FAKE/MISINFO) register the source on-chain."""
    onchain: Dict[str, Any] = {"source_url": source_url}
    if not source_url:
        onchain["skipped"] = "no_source_url"
        return onchain

    if not publisher:
        try:
            parsed = urlparse(source_url)
            publisher = parsed.netloc or None
        except Exception:
            publisher = None
    if publisher:
        onchain["publisher"] = publisher

    try:
        onchain["reputation"] = check_source_reputation(source_url)
    except Exception as e:
        onchain["reputation_error"] = str(e)

    if publisher:
        try:
            onchain["publisher_reputation"] = get_sources_by_publisher(publisher)
        except Exception as e:
            onchain["publisher_reputation_error"] = str(e)

    if classification in {"FAKE", "MISINFORMATION"}:
        try:
            onchain["registration"] = register_flagged_source(
                source_url, publisher=publisher or "FakeNewsDetector"
            )
        except Exception as e:
            onchain["registration_error"] = str(e)

    return onchain

class BengaliSemanticRetriever:
    """Bengali-specific semantic search for news articles"""
    
    def __init__(self, api_key: str, num_results: int = 5):
        self.api_key = api_key
        self.num_results = num_results
        self.bengali_sources = [
            'site:prothomalo.com',
            'site:bdnews24.com', 
            'site:jugantor.com',
            'site:ittefaq.com.bd',
            'site:samakal.com',
            'site:kalerkantho.com',
            'site:amadershomoy.com',
            'site:dailynayadiganta.com',
            'site:manabzamin.com',
            'site:banglanews24.com'
        ]
        
    def search_bengali_news(self, query: str) -> List[Document]:
        """Search for Bengali news articles using direct Serper API"""
        try:
            # Get raw search results from Serper API directly
            raw_results = self._get_bengali_search_results(query)
            if not raw_results:
                return []
            
            # Convert results to documents without semantic processing
            documents = []
            for result in raw_results[:self.num_results]:
                content = f"{result.get('title', '')}\n{result.get('snippet', '')}"
                documents.append(Document(
                    page_content=content, 
                    metadata={
                        "source": result.get("link", ""),
                        "title": result.get('title', ''),
                        "language": "bengali",
                        "date": result.get("date", "")  # Capture publication date if available
                    }
                ))
            
            print(f"Found {len(documents)} Bengali documents")
            return documents
            
        except Exception as e:
            print(f"Bengali search error: {e}")
            return []
    
    def _get_bengali_search_results(self, query: str) -> List[Dict]:
        """Get Bengali news search results"""
        try:
            _SERPER_SEARCH_URL = "https://google.serper.dev/search"
            
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Use the exact same format as your working JavaScript code
            payload = {
                "q": query,  # Use Bengali query directly like in your working example
                "gl": "bd",  # Bangladesh
                "hl": "bn"   # Bengali language
            }
            
            print(f"Searching with Bengali query: {query}")
            
            resp = requests.post(_SERPER_SEARCH_URL, headers=headers, json=payload, timeout=10)
            
            if resp.status_code != 200:
                raise Exception(f"Serper API Error: {resp.text}")
            
            results = resp.json()
            return results.get("organic", [])
            
        except Exception as e:
            print(f"Bengali search error: {e}")
            return []
    
    def _extract_english_keywords(self, bengali_text: str) -> str:
        """Extract English keywords from Bengali text for search"""
        # Common Bengali-English word mappings for news
        keyword_mappings = {
            'সেনাবাহিনী': 'army military',
            'বিরুদ্ধে': 'against',
            'পরিকল্পিত': 'planned',
            'ষড়যন্ত্র': 'conspiracy',
            'রুখতে': 'stop prevent',
            'হবে': 'will',
            'খবর': 'news',
            'রাজনীতি': 'politics',
            'অর্থনীতি': 'economy',
            'শিক্ষা': 'education',
            'স্বাস্থ্য': 'health',
            'ক্রীড়া': 'sports',
            'বিনোদন': 'entertainment',
            'প্রযুক্তি': 'technology',
            'পরিবেশ': 'environment',
            'আইন': 'law',
            'নিরাপত্তা': 'security',
            'দুর্নীতি': 'corruption',
            'চুরি': 'theft',
            'হত্যা': 'murder',
            'দুর্ঘটনা': 'accident',
            'বন্যা': 'flood',
            'ভূমিকম্প': 'earthquake',
            'আগুন': 'fire',
            'বোমা': 'bomb',
            'আত্মঘাতী': 'suicide',
            'সন্ত্রাস': 'terrorism'
        }
        
        keywords = []
        for bengali_word, english_words in keyword_mappings.items():
            if bengali_word in bengali_text:
                keywords.extend(english_words.split())
        
        return ' '.join(keywords) if keywords else ''
    
    def generate_bengali_search_queries(self, claim: str) -> List[str]:
        """Generate Bengali search query variations"""
        try:
            # Create simple Bengali query variations manually to avoid API issues
            queries = []
            
            # Original claim - keep it as is
            queries.append(claim)
            
            # Extract key words from the claim for better search
            # Split by spaces, not by characters
            words = claim.split()
            if len(words) > 1:
                # Use first few words
                if len(words) >= 3:
                    queries.append(' '.join(words[:3]))
                # Use last few words  
                if len(words) >= 3:
                    queries.append(' '.join(words[-3:]))
                # Use middle words
                if len(words) > 4:
                    queries.append(' '.join(words[1:-1]))
            
            # Remove duplicates and empty queries
            unique_queries = []
            for q in queries:
                q = q.strip()
                if q and q not in unique_queries and len(q) > 2:
                    unique_queries.append(q)
            
            print(f"Generated Bengali queries: {unique_queries}")
            return unique_queries[:3]  # Limit to 3 queries
            
        except Exception as e:
            print(f"Error generating Bengali queries: {e}")
            return [claim]  # Fallback to original claim

class SemanticNewsRetriever:
    """Semantic search for news articles using embeddings"""
    
    def __init__(self, api_key: str, num_results: int = 20, embedding_model: SentenceTransformer = None):
        self.api_key = api_key
        self.num_results = num_results
        # Reuse shared embedding model if provided, otherwise create new one
        if embedding_model is not None:
            self.model = embedding_model
        else:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.index = None
        self.documents = []
        
    def search_news(self, query: str) -> List[Document]:
        """Search for news articles using semantic similarity"""
        try:
            # First, get raw search results from Serper
            raw_results = self._get_raw_search_results(query)
            if not raw_results:
                return []
            
            # Create embeddings for the query and documents
            query_embedding = self.model.encode([query])
            doc_texts = [f"{result['title']} {result['snippet']}" for result in raw_results]
            doc_embeddings = self.model.encode(doc_texts)
            
            # Calculate similarities
            similarities = np.dot(query_embedding, doc_embeddings.T).flatten()
            
            # Sort by similarity and get top results
            top_indices = np.argsort(similarities)[::-1][:self.num_results]
            
            documents = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Threshold for relevance
                    result = raw_results[idx]
                    content = f"{result.get('title', '')}\n{result.get('snippet', '')}"
                    documents.append(Document(
                        page_content=content, 
                        metadata={
                            "source": result.get("link", ""),
                            "similarity": float(similarities[idx]),
                            "title": result.get('title', ''),
                            "date": result.get("date", "")  # Capture publication date if available
                        }
                    ))
            
            print(f"Found {len(documents)} semantically relevant documents")
            return documents
            
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def _get_raw_search_results(self, query: str) -> List[Dict]:
        """Get raw search results from Serper API"""
        try:
            _SERPER_SEARCH_URL = "https://google.serper.dev/search"
            
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": f'"{query}" (news OR article OR report) -opinion -editorial',
                "num": 50,  # Get more results for better semantic filtering
            }
            
            resp = requests.post(_SERPER_SEARCH_URL, headers=headers, json=payload, timeout=10)
            
            if resp.status_code != 200:
                raise Exception(f"Serper API Error: {resp.text}")
            
            results = resp.json()
            return results.get("organic", [])
            
        except Exception as e:
            print(f"Raw search error: {e}")
            return []

class SerperRetrieverWrapper:
    #Class to use Serper as retriver agent for the RAG framework
    def __init__(self, api_key: str, num_results: int = 15):
        self.api_key = api_key
        self.num_results = num_results
    
    def get_relevant_documents(self, query: str):
        """
        Query Serper.dev and return up to `num_results` organic search hits.
        Each hit is a dict: { "title": str, "link": str, "snippet": str }.
        """
        try:
            _SERPER_SEARCH_URL = "https://google.serper.dev/search"
    
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": f'"{query}" (news OR article OR report) -opinion -editorial',
                "num": self.num_results,
            }
            
            resp = requests.post(_SERPER_SEARCH_URL, headers=headers, json=payload, timeout=5)
            
            if resp.status_code != 200:
                raise Exception(f"Serper API Error: {resp.text}")
            results = resp.json()
            
            documents = []
            for result in results.get("organic", [])[:self.num_results]:
                content = f"{result.get('title', '')}\n{result.get('snippet', '')}"
                documents.append(Document(
                    page_content=content, 
                    metadata={
                        "source": result.get("link", ""),
                        "title": result.get('title', ''),
                        "date": result.get("date", "")  # Capture publication date if available
                    }
                ))
            
            if not documents:
                empty_search = Exception("No result found")
                raise empty_search
            
            return documents
        except Exception as e:
            if e is empty_search:
                print('serper main exception1')
                raise empty_search
            else:
                print('serper main exception2')
                raise e

def _extract_text_from_html(html_content: str, max_length: int = 1500) -> str:
    """
    Convert HTML to plain text by dropping script/style tags and stripping markup.
    Truncates to avoid oversized context.
    """
    if not html_content:
        return ""

def _extract_title_from_html(html_content: str) -> str:
    """
    Pull page title from HTML. Returns empty string if not found.
    """
    if not html_content:
        return ""
    try:
        match = TITLE_PATTERN.search(html_content)
        if match:
            raw_title = html.unescape(match.group(1))
            return ' '.join(raw_title.split())
    except Exception as e:
        print(f"Title extraction error: {e}")
    return ""

def _extract_meta_description(html_content: str) -> str:
    """
    Extract meta description content if available.
    """
    if not html_content:
        return ""
    try:
        desc_match = re.search(
            r'<meta[^>]+name=["\\\']description["\\\'][^>]*content=["\\\'](.*?)["\\\']',
            html_content,
            flags=re.IGNORECASE | re.DOTALL
        )
        if desc_match:
            raw_desc = html.unescape(desc_match.group(1))
            return ' '.join(raw_desc.split())
    except Exception as e:
        print(f"Meta description extraction error: {e}")
    return ""

def _keyword_overlap(a: str, b: str) -> float:
    """Simple token overlap ratio between two strings."""
    if not a or not b:
        return 0.0
    tokens_a = {t for t in re.findall(r"[\\w]+", a.lower()) if len(t) > 3}
    tokens_b = {t for t in re.findall(r"[\\w]+", b.lower()) if len(t) > 3}
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a.intersection(tokens_b)
    return len(inter) / max(len(tokens_a), 1)

def _is_refuting(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(k in lower for k in REFUTE_KEYWORDS)

def _compute_flagged_sources(claim_text: str, documents: List[Document], embedding_model: SentenceTransformer, sim_threshold: float = 0.7) -> List[Dict]:
    """Flag sources that closely match a fake/misinfo claim."""
    if not claim_text or not documents or embedding_model is None:
        return []
    valid_docs = []
    doc_texts = []
    for doc in documents:
        source = (doc.metadata or {}).get("source", "")
        if source in {"submitted_article", "submitted_url"}:
            continue
        title = (doc.metadata or {}).get("title", "")
        content = (doc.page_content or "").strip()
        if not content and not title:
            continue
        combined = (title + "\n" + content).strip()
        if not combined:
            continue
        valid_docs.append((doc, combined, title, content))
        doc_texts.append(combined)

    if not valid_docs:
        return []

    try:
        claim_vec = embedding_model.encode([claim_text])[0]
        doc_vecs = embedding_model.encode(doc_texts)
    except Exception as e:
        print(f"Error computing similarity for flagged sources: {e}")
        return []

    flagged = []
    for idx, (doc, combined, title, content) in enumerate(valid_docs):
        doc_vec = doc_vecs[idx]
        sim = float(np.dot(claim_vec, doc_vec) / (np.linalg.norm(claim_vec) * np.linalg.norm(doc_vec) + 1e-8))
        overlap = _keyword_overlap(claim_text, combined)
        stance = "refute" if _is_refuting(combined) else "assert"
        if stance == "refute":
            continue
        if sim >= sim_threshold and (overlap >= 0.2 or sim >= 0.8):
            snippet = content[:240] if content else combined[:240]
            flagged.append({
                "url": (doc.metadata or {}).get("source", ""),
                "title": title,
                "snippet": snippet,
                "similarity": round(sim, 3),
                "overlap": round(overlap, 3),
                "stance": stance,
                "reason": "High similarity match to flagged claim"
            })
    return flagged
    try:
        # Remove script and style blocks
        cleaned = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        # Strip all remaining tags
        cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
        cleaned = html.unescape(cleaned)
        # Collapse whitespace
        cleaned = ' '.join(cleaned.split())
        if len(cleaned) > max_length:
            return cleaned[:max_length]
        return cleaned
    except Exception as e:
        print(f"HTML extraction error: {e}")
        return ""

def _fetch_url_snapshot(url: str, timeout: int = 10) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch URL content and return raw HTML plus a cleaned text version.
    """
    if not url:
        return None, None
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FactCheckBot/1.0; +https://example.com/bot)"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(f"URL fetch failed ({resp.status_code}) for {url}")
            return None, None
        html_content = resp.text
        text_content = _extract_text_from_html(html_content, max_length=1500)
        return html_content, text_content
    except Exception as e:
        print(f"URL fetch error for {url}: {e}")
        return None, None

def _extract_evidence_from_documents(documents: List[Document]) -> List[Dict]:
    """
    Extract evidence sources from document list
    
    Args:
        documents: List of Document objects with metadata
        
    Returns:
        List of dicts with url, title, snippet, and date (limited to top 10)
    """
    evidence_sources = []
    for doc in documents[:10]:  # Limit to top 10 sources
        evidence_sources.append({
            "url": doc.metadata.get("source", ""),
            "title": doc.metadata.get("title", ""),
            "snippet": doc.page_content[:200] if doc.page_content else "",  # First 200 chars
            "date": doc.metadata.get("date", "")  # Publication date if available
        })
    return evidence_sources
        
def verify_news(user_claim, submitted_url: str = None, input_lang = 'auto'):
    """
       Description: This function is used to verify the claim provided by the user and output as REAL or FAKE or UNSURE based on the context with a short explanation
       INPUT: user_claim --> The news user wish to verify
       OUTPUT: (claim, verdict_orig, verdict_trans, credibility_score) - 4-tuple with credibility score (0-100 for FAKE, 0 otherwise)
    """
    try:
       print('here1: ', user_claim)

       # If claim text is missing OR is just the URL, fetch and build claim from page content
       url_only_input = submitted_url and user_claim and user_claim.strip() == submitted_url.strip()
       if ((not user_claim or not user_claim.strip()) or url_only_input) and submitted_url:
           url_raw_html, url_text_content = _fetch_url_snapshot(submitted_url)
           if url_raw_html is None:
               error_msg = 'URL fetch failed. Please try again after some time.'
               return (error_msg, error_msg, error_msg, 0, 'UNSURE', [])

           # Fallbacks: try meta description if cleaned body is empty
           if not url_text_content:
               meta_desc = _extract_meta_description(url_raw_html)
               page_title_only = _extract_title_from_html(url_raw_html)
               if meta_desc:
                   url_text_content = meta_desc
               elif page_title_only:
                   url_text_content = page_title_only
               else:
                   error_msg = 'The fetched article seems empty. Please provide a different link.'
                   return (error_msg, error_msg, error_msg, 0, 'UNSURE', [])

           page_title = _extract_title_from_html(url_raw_html)
           combined_claim = f"{page_title}\n\n{url_text_content}".strip()
           user_claim = combined_claim if combined_claim else url_text_content

       claim1 = user_claim.replace("'","")
       claim1 = claim1.replace("\n"," ")
       print('here2: ', claim1)
       
       # Detect input language
       is_bengali = any('\u0980' <= char <= '\u09FF' for char in claim1)
       
       if is_bengali:
           print("Detected Bengali input - using Bengali-specific search")
           claim, verdict_orig, verdict_trans, credibility_score, claim_id, classification, evidence_sources, onchain_metadata = verify_bengali_news(
               claim1,
               submitted_url=submitted_url,
               article_text=user_claim
           )
       else:
           print("Non-Bengali input - using English search with translation")
           claim, verdict_orig, verdict_trans, credibility_score, claim_id, classification, evidence_sources, onchain_metadata = verify_english_news(
               claim1,
               submitted_url=submitted_url,
               article_text=user_claim,
               input_lang=input_lang
           )
       return (claim, verdict_orig, verdict_trans, credibility_score, classification, evidence_sources, onchain_metadata)
    except Exception as e:
        print('Error in main verification: ', e)
        error_msg = 'Something went wrong. Please try after some time'
        return(error_msg, error_msg, error_msg, 0, 'UNSURE', [], {})

def verify_bengali_news(claim: str, submitted_url: str = None, article_text: str = None):
    """
    Verify Bengali news claims using Bengali-specific search
    Returns: (claim, verdict_orig, verdict_trans, credibility_score, claim_id) - 5-tuple
    """
    try:
        # Use Bengali semantic retriever
        bengali_retriever = BengaliSemanticRetriever(api_key=serp_dev_api_key, num_results=5)
        
        # Generate Bengali search queries
        queries = bengali_retriever.generate_bengali_search_queries(claim)
        
        # Search directly with the generated queries instead of using RunnableLambda
        all_documents = []
        print(f"Generated Bengali search queries: {queries}")
        
        url_raw_html, url_text_content = _fetch_url_snapshot(submitted_url) if submitted_url else (None, None)

        for query in queries:
            try:
                docs = bengali_retriever.search_bengali_news(query.strip())
                print(f"Found {len(docs)} Bengali results for query: {query}")
                all_documents.extend(docs)
            except Exception as e:
                print(f"Bengali search failed for query '{query}': {e}")
                continue
        
        # Remove duplicates based on source URL
        seen_sources = set()
        unique_docs = []
        for doc in all_documents:
            source = doc.metadata.get('source', '')
            if source not in seen_sources:
                seen_sources.add(source)
                unique_docs.append(doc)

        # Add submitted article text and fetched URL content as context documents
        if article_text:
            unique_docs.insert(0, Document(
                page_content=article_text[:4000],
                metadata={"source": "submitted_article", "title": "Submitted Article Text", "language": "bengali"}
            ))
        if url_text_content:
            url_text_content = url_text_content[:1200]
            unique_docs.insert(0, Document(
                page_content=url_text_content,
                metadata={"source": submitted_url or "submitted_url", "title": "Fetched URL Content", "language": "bengali"}
            ))
        
        print(f"Total unique Bengali documents found: {len(unique_docs)}")
        
        # Limit to top 5 documents to stay within token limits
        limited_docs = unique_docs[:5]
        print(f"Using {len(limited_docs)} documents for processing")
        
        # Create a simple context retriever that returns the documents
        def simple_context_retriever(input_data):
            return limited_docs
        
        context_retriever = RunnableLambda(simple_context_retriever)
        
        # Bengali-specific summarizer
        bengali_summarizer_template = '''
           তুমি একজন বাংলা ভাষার বিশেষজ্ঞ। তোমার কাজ হলো খবরের নিবন্ধ থেকে মূল ঘটনা বের করা।
       
           দাবি: {question}
           
           নিবন্ধসমূহ:
           {context}
           
           শুধুমাত্র ঘটনামূলক তথ্য বের করো। সংক্ষিপ্ত সারসংক্ষেপ দাও।
        '''
        
        summarizer_prompt = PromptTemplate.from_template(bengali_summarizer_template)
        llm_summarizer = ChatGroq(api_key = grok_api_key, model_name = model_summarizer)
        
        summarizer_chain = (
            {
                "context": context_retriever,
                "question": RunnablePassthrough()
            }
            | summarizer_prompt
            | llm_summarizer
            | StrOutputParser()
        )
        
        # Bengali fact-checker
        bengali_fact_checker_template = '''
           তুমি একজন সত্যতা যাচাইকারী সহায়ক।
           
           দাবি: {question}
           
           প্রমাণ:
           {evidence}
           
           নির্দেশনা:
           - শক্ত প্রমাণ পেলে REAL
           - বিপরীত বা বিভ্রান্তিকর/মিথ্যা তথ্য হলে FAKE
           - ইচ্ছাকৃত বা অনিচ্ছাকৃত ভ্রান্ত/অর্ধসত্য তথ্য হলে MISINFORMATION
           - অপর্যাপ্ত প্রমাণ পেলে UNSURE
           - REAL/FAKE/MISINFORMATION এর জন্য 0-100 স্কেল এ বিশ্বাসযোগ্যতা স্কোর দাও (100 = সম্পূর্ণ নিশ্চিত)
           - যদি REAL হয়, ব্যাখ্যায় সমর্থনকারী প্রমাণের সংক্ষিপ্ত সারসংক্ষেপ অন্তর্ভুক্ত করুন
           
           উত্তর:
           Classification: REAL or FAKE or MISINFORMATION or UNSURE
           Credibility Score: <0-100 for REAL or FAKE or MISINFORMATION, N/A for UNSURE>
           Explanation: <সংক্ষিপ্ত যুক্তি>
        '''
        
        fact_checker_prompt = PromptTemplate.from_template(bengali_fact_checker_template)
        llm_fact_checker = ChatGroq(api_key = grok_api_key, model_name = model_judge)
        
        fact_checker_chain = (
            {
                "question": RunnablePassthrough(),
                "evidence": summarizer_chain 
            }
            | fact_checker_prompt
            | llm_fact_checker
            | StrOutputParser()
        )
        
        verdict_orig = fact_checker_chain.invoke(claim)
        
        # Parse verdict output to extract classification, credibility score, and explanation
        lines = verdict_orig.split('\n')
        verdict_class = lines[0] if len(lines) > 0 else "Classification: UNSURE"
        credibility_score = 0
        
        # Extract credibility score using helper
        for line in lines:
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in SCORE_KEYWORDS_EN) or any(keyword in line for keyword in SCORE_KEYWORDS_BN):
                credibility_score = _extract_score_from_text(line)
                break
        if credibility_score == 0:
            credibility_score = _extract_score_from_text(verdict_orig)
        
        # Extract explanation (everything after classification and score)
        explanation_lines = []
        found_class = False
        found_score = False
        for line in lines:
            if 'Classification:' in line:
                found_class = True
                continue
            if 'Credibility Score:' in line or 'credibility' in line.lower():
                found_score = True
                continue
            if found_class and (found_score or 'Explanation:' in line or len(explanation_lines) > 0):
                if 'Explanation:' in line:
                    explanation_lines.append(line.replace('Explanation:', '').strip())
                else:
                    explanation_lines.append(line.strip())
        
        verdict_explan = '\n'.join(explanation_lines) if explanation_lines else verdict_orig.split('\n')[-1]
        
        # Extract classification text
        classification_text = verdict_class.replace('Classification', '').replace(':', '').strip()
        classification = _infer_classification(classification_text)
        
        # Only set credibility score to 0 for UNSURE (others keep their scores)
        if classification == 'UNSURE':
            credibility_score = 0
        
        verdict_orig = verdict_class + '\n\n' + verdict_explan
        verdict_trans = verdict_orig  # Already in Bengali
        
        # Extract evidence sources and save for all classifications
        claim_id = None
        evidence_sources = _extract_evidence_from_documents(limited_docs)
        flagged_sources = []
        if classification in {'FAKE', 'MISINFORMATION'}:
            flagged_sources = _compute_flagged_sources(claim, limited_docs, shared_embedding_model)
        source_identifier = submitted_url or claim
        onchain_metadata = _build_onchain_metadata(
            source_identifier,
            publisher=None,
            classification=classification
        )
        try:
            claim_id = claim_storage.save_claim_record(
                claim_text=claim,
                claim_text_original=claim,
                classification=classification,
                credibility_score=credibility_score,
                explanation=verdict_explan,
                evidence_sources=evidence_sources,
                language='bengali',
                submitted_url=submitted_url,
                article_text=article_text or claim,
                url_snapshot_html=url_raw_html,
                flagged_sources=flagged_sources,
                onchain_metadata=onchain_metadata
            )
        except Exception as storage_error:
            print(f"Error saving claim to storage: {storage_error}")
            # Continue even if storage fails
        
        return(claim, verdict_orig, verdict_trans, credibility_score, claim_id, classification, evidence_sources, onchain_metadata)
        
    except Exception as e:
        print('Error in Bengali verification: ', e)
        error_msg = 'কিছু সমস্যা হয়েছে। দয়া করে আবার চেষ্টা করুন'
        return(error_msg, error_msg, error_msg, 0, None, 'UNSURE', [])

def verify_english_news(claim: str, submitted_url: str = None, article_text: str = None, input_lang = 'auto'):
    """
    Verify non-Bengali news claims using English search with translation
    Returns: (claim, verdict_orig, verdict_trans, credibility_score, claim_id) - 5-tuple
    """
    try:
       print('here1: ', claim)
       claim1 = claim.replace("'","")
       claim1 = claim.replace("\n"," ")
       print('here2: ', claim1)
        
       # Use semantic search instead of keyword search (with shared embedding model)
       semantic_retriever = SemanticNewsRetriever(api_key=serp_dev_api_key, num_results=20, embedding_model=shared_embedding_model)
       
       # Store documents reference for evidence extraction
       retrieved_documents = []
       url_raw_html, url_text_content = _fetch_url_snapshot(submitted_url) if submitted_url else (None, None)
       submitted_article_text = article_text or claim1
       
       def semantic_search_multiple_queries(queries):
           """Search for multiple query variations using semantic similarity"""
           nonlocal retrieved_documents
           all_documents = []
           print(f"Generated search queries: {queries}")
           
           # Fallback to old search method if semantic search fails
           fallback_retriever = SerperRetrieverWrapper(api_key=serp_dev_api_key)
           
           for query in queries:
               try:
                   docs = semantic_retriever.search_news(query.strip())
                   print(f"Found {len(docs)} semantically relevant results for query: {query}")
                   all_documents.extend(docs)
               except Exception as e:
                   print(f"Semantic search failed for query '{query}': {e}")
                   # Fallback to keyword search
                   try:
                       docs = fallback_retriever.get_relevant_documents(query.strip())
                       print(f"Fallback: Found {len(docs)} keyword-based results for query: {query}")
                       all_documents.extend(docs)
                   except Exception as fallback_e:
                       print(f"Both semantic and keyword search failed for query '{query}': {fallback_e}")
                       continue
           
           # Remove duplicates based on source URL
           seen_sources = set()
           unique_docs = []
           for doc in all_documents:
               source = doc.metadata.get('source', '')
               if source not in seen_sources:
                   seen_sources.add(source)
                   unique_docs.append(doc)
           
           # Inject submitted content for context
           if submitted_article_text:
               unique_docs.insert(0, Document(
                   page_content=submitted_article_text[:4000],
                   metadata={"source": "submitted_article", "title": "Submitted Article Text"}
               ))
           if url_text_content:
               url_text_content = url_text_content[:1200]
               unique_docs.insert(0, Document(
                   page_content=url_text_content,
                   metadata={"source": submitted_url or "submitted_url", "title": "Fetched URL Content"}
               ))
           
           print(f"Total unique documents found (including submitted content): {len(unique_docs)}")
           retrieved_documents = unique_docs[:20]  # Store for evidence extraction
           return unique_docs[:20]  # Limit to top 20 results
       
       context_retriever = RunnableLambda(semantic_search_multiple_queries)
       
       #Multi Query Generation
       multi_query_template = """You are an AI language model assistant. Your task is to generate three 
       different search queries for fact-checking the given news claim. Create variations that will help 
       find relevant news articles and reports about this topic. Focus on different aspects like:
       1. The main event/claim
       2. Key people or organizations mentioned
       3. Location or time period if relevant
       
       Provide these alternative search queries separated by newlines. Original claim: {question}"""
       perspectives_prompt = ChatPromptTemplate.from_template(multi_query_template)
       
       llm_multi_query = ChatGroq(api_key = grok_api_key, model_name = model_multi_query)
       
       generate_queries = (
           perspectives_prompt 
           | llm_multi_query
           | StrOutputParser() 
           | (lambda x: x.split("\n"))
       )
       
       #Summarization using multi query
       summarizer_template = '''
          You are an assistant summarizing factual evidence from multiple news articles.
       
          Based on the following documents, extract the key facts relevant to the claim.
          Focus on factual information that directly supports or contradicts the claim.
          
          Claim: {question}
          
          Documents:
          {context}
          
          Instructions:
          - Extract only factual information, not opinions
          - Note the source and date when available
          - Identify any contradictions between sources
          - Focus on verifiable facts related to the claim
          
          Return a concise summary of the key facts found.
       '''
       summarizer_prompt = PromptTemplate.from_template(summarizer_template)
       
       llm_summarizer = ChatGroq(api_key = grok_api_key, model_name = model_summarizer)
       summarizer_chain = (
           {
               "context": context_retriever,
               "question": generate_queries
           }
           | summarizer_prompt
           | llm_summarizer
           | StrOutputParser()
       )
       
       #Final Judgement 
       fact_checker_template = '''
          You are a fact-checking assistant. Your job is to determine if a news claim is accurate based on available evidence.
          
          Claim: {question}
          
          Evidence:
          {evidence}
          
          Instructions:
          - If you find strong evidence supporting the claim from reliable news sources, classify as REAL
          - If you find evidence contradicting the claim from reliable sources, classify as FAKE  
          - If the claim contains misleading, incomplete, or out-of-context information, classify as MISINFORMATION
          - If the evidence is insufficient, unclear, or conflicting, classify as UNSURE
          - Consider the credibility and recency of sources
          - Be conservative - when in doubt, choose UNSURE rather than making assumptions
          - If you classify as REAL, FAKE, or MISINFORMATION, provide a credibility score (0-100) indicating confidence (100 = completely certain)
          - For REAL classifications, summarize the key supporting evidence from reliable sources
          
          Respond in this format:
          Classification: REAL or FAKE or MISINFORMATION or UNSURE
          Credibility Score: <0-100 for REAL or FAKE or MISINFORMATION, N/A for UNSURE>
          Explanation: <your detailed reasoning based on the evidence>
       '''
       
       fact_checker_prompt = PromptTemplate.from_template(fact_checker_template)
       
       llm_fact_checker = ChatGroq(api_key = grok_api_key, model_name = model_judge)
       fact_checker_chain = (
           {
               "question": RunnablePassthrough(),
               "evidence": summarizer_chain 
           }
           | fact_checker_prompt
           | llm_fact_checker
           | StrOutputParser()
       )
   
       claim = claim1
       
       #Calling SARVAM API to translate Indic languages to English
       client = SarvamAI(api_subscription_key = sarvam_api_key)
       
       try:
           translation = client.text.translate(
           input=claim,
           source_language_code="auto",
           target_language_code="en-IN"
           )
       except Exception as e:
           print(f"Error during translation: {e}")
           error_msg = 'It appears you have provided input in an alien language. Please try again with some other language'
           return error_msg, error_msg, error_msg, 0, None
       
       claim_final = translation.translated_text if translation else claim
       claim_orig_lang = translation.source_language_code
       print(f"Translated claim: {claim_final}")
       
       verdict_orig = fact_checker_chain.invoke(claim_final)
       
       # Parse verdict output to extract classification, credibility score, and explanation
       lines = verdict_orig.split('\n')
       verdict_class = lines[0] if len(lines) > 0 else "Classification: UNSURE"
       credibility_score = 0
       
       # Extract credibility score using regex
       for line in lines:
           lower_line = line.lower()
           if any(keyword in lower_line for keyword in SCORE_KEYWORDS_EN) or any(keyword in line for keyword in SCORE_KEYWORDS_BN):
               credibility_score = _extract_score_from_text(line)
               break
       if credibility_score == 0:
           credibility_score = _extract_score_from_text(verdict_orig)
       
       # Extract explanation (everything after classification and score)
       explanation_lines = []
       found_class = False
       found_score = False
       for line in lines:
           if 'Classification:' in line:
               found_class = True
               continue
           if 'Credibility Score:' in line or 'credibility' in line.lower():
               found_score = True
               continue
           if found_class and (found_score or 'Explanation:' in line or len(explanation_lines) > 0):
               if 'Explanation:' in line:
                   explanation_lines.append(line.replace('Explanation:', '').strip())
               else:
                   explanation_lines.append(line.strip())
       
       verdict_explan = '\n'.join(explanation_lines) if explanation_lines else verdict_orig.split('\n')[-1]
       
       # Extract classification text
       classification_text = verdict_class.replace('Classification', '').replace(':', '').strip()
       classification = _infer_classification(classification_text)
       
       # Only set credibility score to 0 for UNSURE (others keep their scores)
       if classification == 'UNSURE':
           credibility_score = 0
       
       if input_lang == 'auto':
           trans_lang = claim_orig_lang
       else:
           trans_lang = input_lang
   
       if claim_orig_lang != 'en-IN':
           try:
               translation_class = client.text.translate(
               input=verdict_class,
               source_language_code='en-IN',
               target_language_code=claim_orig_lang
               )
           except Exception as e:
               print(f"Error during verdict translation: {e}")  
               error_msg = 'Something went wrong while translating the verdict. Please try again'
               return error_msg, error_msg, error_msg, 0, None
               
           try:
               translation_explan = client.text.translate(
               input=verdict_explan,
               source_language_code='en-IN',
               target_language_code=claim_orig_lang
               )
           except Exception as e:
               print(f"Error during verdict translation: {e}")  
               error_msg = 'Something went wrong while translating the verdict. Please try again'
               return error_msg, error_msg, error_msg, 0, None
           
           verdict_trans_class = translation_class.translated_text
           verdict_trans_explan = translation_explan.translated_text
           verdict_trans = verdict_trans_class + '\n\n' + verdict_trans_explan
           
           verdict_orig = verdict_class + '\n\n' + verdict_explan
       else:
           verdict_orig = verdict_class + '\n\n' + verdict_explan
           verdict_trans = verdict_orig
       
       # Extract evidence sources and save for all classifications
       claim_id = None
       evidence_sources = _extract_evidence_from_documents(retrieved_documents)
       flagged_sources = []
       if classification in {'FAKE', 'MISINFORMATION'}:
           flagged_sources = _compute_flagged_sources(claim_final, retrieved_documents, shared_embedding_model)
       source_identifier = submitted_url or claim_final
       onchain_metadata = _build_onchain_metadata(
           source_identifier,
           publisher=None,
           classification=classification
       )
       try:
           claim_id = claim_storage.save_claim_record(
               claim_text=claim_final,
               claim_text_original=claim,
               classification=classification,
               credibility_score=credibility_score,
               explanation=verdict_explan,
               evidence_sources=evidence_sources,
               language=claim_orig_lang,
               submitted_url=submitted_url,
               article_text=article_text or claim,
               url_snapshot_html=url_raw_html,
               flagged_sources=flagged_sources,
               onchain_metadata=onchain_metadata
           )
       except Exception as storage_error:
           print(f"Error saving claim to storage: {storage_error}")
           # Continue even if storage fails
       
       return(claim_final, verdict_orig, verdict_trans, credibility_score, claim_id, classification, evidence_sources, onchain_metadata)
    except Exception as e:
        if str(e) == 'No result found':
            print('Error in main proc1. Error is ', e)
            error_msg = 'The search for this claim came back empty. Please rephrase the claim or try with a new one'
            return('UNSURE' ,error_msg,error_msg,0, None, 'UNSURE', [], {})
        else:
            print('Error in main proc2. Error is ', e)
            error_msg = 'Something went wrong. Please try after some time'
            return(error_msg,error_msg,error_msg,0, None, 'UNSURE', [], {})
    
def transcribe_audio(audio):
    """
       Description: This function trascibes audio using SarvamAI STT model 
    """
    try:
        client = SarvamAI(api_subscription_key = sarvam_api_key)
        mime_type, _ = mimetypes.guess_type(audio)
        
        with open(audio, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=("audio.mp3", f, mime_type or "audio/mpeg"),
                model="saarika:v2.5",
                language_code="unknown"
            )
        ret_var = response.transcript
        ret_lang = response.language_code
    except Exception as e:
        print(f"Error during translation: {e}")
        ret_var = ''

    return ret_var, ret_lang
    
def verify_news_audio(audio):
    """
       Description: This function verifies the news where input method is Audio
    """
    
    claim, orig_lang = transcribe_audio(audio)
    if claim == '':
        error_msg = 'I could not understand your message. Please try recording again'
        return(error_msg,error_msg,error_msg)
    
    final_claim, verdict, verdict_trans, _, _, evidence_sources, _ = verify_news(claim, input_lang=orig_lang)
    return final_claim, verdict, verdict_trans
    
if __name__ == '__main__':
    print('helllo')
