"""Semantic search engine with semantic-first search strategy - OPTIMIZED."""

import re
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.models import Video, TranscriptSegment, Embedding, SearchQuery
from embeddings.text_embeddings import get_embedding_generator
from search.reranker import get_reranker
from search.norwegian_stemmer import stem_matches_in_text

# Global vocabulary cache (built once from transcript data)
_vocabulary: Optional[Set[str]] = None

# ── Stop words and command verbs (shared across all search methods) ──
# English stop words
_STOP_WORDS_EN = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "it",
    "as",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "can",
    "could",
    "not",
    "no",
    "nor",
    "so",
    "if",
    "then",
    "than",
    "that",
    "this",
    "these",
    "those",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "when",
    "where",
    "why",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "some",
    "any",
    "other",
    "into",
    "about",
    "between",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "up",
    "down",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "once",
    "here",
    "there",
    "very",
    "just",
    "also",
    "too",
    "only",
    "own",
    "same",
    "such",
    # Command verbs (natural-language queries)
    "tell",
    "me",
    "show",
    "give",
    "let",
    "please",
    "find",
    "get",
    "list",
    "display",
    "search",
    "look",
    "see",
    "want",
    "need",
}

# Norwegian (Bokmål + Nynorsk) stop words
_STOP_WORDS_NO = {
    "og", "i", "jeg", "det", "at", "en", "et", "den", "til", "er",
    "som", "på", "de", "med", "han", "av", "ikke", "der", "så",
    "var", "meg", "seg", "men", "ett", "har", "om", "vi", "min",
    "mitt", "ha", "hadde", "hun", "nå", "over", "da", "ved", "fra",
    "du", "ut", "sin", "dem", "oss", "opp", "man", "kan", "hans",
    "hvor", "eller", "hva", "skal", "selv", "sjøl", "her", "alle",
    "vil", "bli", "ble", "blitt", "kunne", "inn", "når", "være",
    "kom", "noen", "noe", "ville", "dere", "denne", "dette", "mitt",
    "også", "under", "etter", "mange", "enn", "ingen", "mot",
    "bli", "bare", "gå", "nå", "mellom", "før", "helt", "andre",
    "fordi", "henne", "hennes", "sitt", "noen", "uten",
    # Command verbs (Norwegian)
    "vis", "finn", "søk", "hent", "vennligst",
}

# Combined stop words (both languages)
STOP_WORDS = _STOP_WORDS_EN | _STOP_WORDS_NO

ANALYTICS_HINT_WORDS = {
    "data",
    "dashboard",
    "kpi",
    "metric",
    "metrics",
    "analytics",
    "analysis",
    "breakdown",
    "report",
    "reports",
    "insight",
    "insights",
    "trend",
    "trends",
    "drilldown",
}


def extract_keywords(query: str) -> List[str]:
    """Extract content-bearing keywords from a query, removing stop words."""
    words = re.findall(r"[\w]+", query.lower(), flags=re.UNICODE)
    return [w for w in words if w not in STOP_WORDS and len(w) >= 2]


def _whole_word_in_text(word: str, text: str) -> bool:
    """True if `word` appears as a whole token (word-boundary match) in `text`.
    Uses Unicode-aware boundaries so that æ, ø, å are treated as word characters."""
    if not word or not text:
        return False
    try:
        # (?<![\w]) and (?![\w]) are Unicode-aware word boundaries
        pattern = rf"(?<![\w]){re.escape(word.lower())}(?![\w])"
        return re.search(pattern, text.lower(), flags=re.UNICODE) is not None
    except re.error:
        return word.lower() in text.lower()


def _is_analytics_intent(query: str) -> bool:
    q = (query or "").lower()
    if "drill down" in q or "drill-down" in q:
        return True
    keywords = extract_keywords(q)
    return any(k in ANALYTICS_HINT_WORDS for k in keywords)


def _is_drill_family(query: str) -> bool:
    q = (query or "").lower()
    return bool(re.search(r"\bdrill(?:ing)?\b", q))


def _facet_suggestions_for_query(query: str) -> List[Dict]:
    """Facet chips for ambiguous queries (returned to the frontend)."""
    if not _is_drill_family(query):
        return []

    return [
        {
            "id": "auto",
            "label": "All meanings",
            "description": "Blend results across meanings (recommended for ambiguous queries).",
        },
        {
            "id": "oil_gas",
            "label": "Oil & gas",
            "description": "Wells, rigs, offshore drilling operations.",
        },
        {
            "id": "tools",
            "label": "Tools",
            "description": "Drilling tools, bits, torque, toolstrings.",
        },
        {
            "id": "analytics",
            "label": "Data / drill-down",
            "description": "Analytics meaning: drill down into data, KPIs, dashboards.",
        },
    ]


# ── Word-sense disambiguation dictionary ──────────────────────────────
# Maps ambiguous words to a list of possible meanings.
# Each meaning has: label (short chip text), phrase (search expansion),
# description (tooltip).  The frontend renders these as clickable chips
# similar to facet chips so the user can narrow the intent.
_AMBIGUOUS_WORDS: Dict[str, List[Dict]] = {
    "well": [
        {"label": "Oil & gas well", "phrase": "oil and gas well", "description": "Wellhead, wellbore, well pressure, well operations"},
        {"label": "Well (adverb)", "phrase": "well", "description": "General adverb usage — 'well done', 'as well as'"},
    ],
    "pipe": [
        {"label": "Pipe / pipeline (oil & gas)", "phrase": "pipe pipeline oil gas", "description": "Flowline, subsea pipe, pipeline installation"},
        {"label": "Pipe (computing)", "phrase": "pipe data stream", "description": "Data pipe, pipeline, streaming"},
    ],
    "platform": [
        {"label": "Offshore platform", "phrase": "offshore platform oil gas", "description": "Fixed or floating offshore structures"},
        {"label": "Software platform", "phrase": "software platform system", "description": "Technology or software platform"},
    ],
    "string": [
        {"label": "Drill string", "phrase": "drill string toolstring", "description": "Drill pipe, BHA, tool string"},
        {"label": "Text string", "phrase": "string text data", "description": "Character string, programming"},
    ],
    "log": [
        {"label": "Well log", "phrase": "well log logging wireline", "description": "Wireline log, mud log, well logging"},
        {"label": "System log", "phrase": "log file system data", "description": "Server log, activity log"},
    ],
    "casing": [
        {"label": "Well casing", "phrase": "well casing cement", "description": "Casing string, casing shoe, cement"},
        {"label": "Casing (enclosure)", "phrase": "casing housing enclosure", "description": "Equipment casing or housing"},
    ],
    "head": [
        {"label": "Wellhead", "phrase": "wellhead christmas tree", "description": "Wellhead, Xmas tree, production head"},
        {"label": "Head (general)", "phrase": "head", "description": "General usage — 'head of department', etc."},
    ],
    "jacket": [
        {"label": "Platform jacket", "phrase": "jacket structure offshore platform", "description": "Offshore jacket structure, substructure"},
        {"label": "Jacket (clothing)", "phrase": "jacket clothing", "description": "Clothing item"},
    ],
    "mud": [
        {"label": "Drilling mud", "phrase": "drilling mud fluid", "description": "Drilling fluid, mud weight, mud pump"},
        {"label": "Mud (general)", "phrase": "mud dirt", "description": "Dirt, soil"},
    ],
    "trip": [
        {"label": "Tripping pipe", "phrase": "trip pipe tripping", "description": "Tripping in/out, round trip"},
        {"label": "Trip (travel)", "phrase": "trip travel journey", "description": "Travel or journey"},
    ],
    "set": [
        {"label": "Set casing / cement", "phrase": "set casing cement plug", "description": "Set casing, cement setting, plug"},
        {"label": "Set (general)", "phrase": "set configure", "description": "General usage — settings, configure"},
    ],
    "flow": [
        {"label": "Flow (oil & gas)", "phrase": "flow rate production oil gas", "description": "Production flow, flow rate, flowline"},
        {"label": "Workflow / data flow", "phrase": "workflow data flow process", "description": "Process flow, workflow, data flow"},
    ],
    "pressure": [
        {"label": "Well pressure", "phrase": "well pressure downhole", "description": "Downhole pressure, BHP, wellbore pressure"},
        {"label": "Pressure (general)", "phrase": "pressure", "description": "General usage"},
    ],
    "tree": [
        {"label": "Christmas tree (wellhead)", "phrase": "christmas tree wellhead subsea", "description": "Subsea or surface Xmas tree"},
        {"label": "Tree (general)", "phrase": "tree", "description": "General usage — decision tree, data tree, etc."},
    ],
    "seal": [
        {"label": "Seal (oil & gas)", "phrase": "seal packer BOP", "description": "BOP seal, packer seal, annular seal"},
        {"label": "Seal (general)", "phrase": "seal", "description": "General usage"},
    ],
    "cap": [
        {"label": "Cap rock / well cap", "phrase": "cap rock well", "description": "Caprock, well cap, capping"},
        {"label": "Cap (general)", "phrase": "cap", "description": "General usage"},
    ],
    "rig": [
        {"label": "Drilling rig", "phrase": "drilling rig offshore", "description": "Drilling rig, derrick, drillship"},
        {"label": "Rig (general)", "phrase": "rig setup", "description": "To rig up, set up"},
    ],
}


def _sense_suggestions(query: str) -> List[Dict]:
    """Return word-sense suggestions for short/ambiguous queries.

    Only triggers when the cleaned query (after stop-word removal) has 1-2
    content words and at least one of them appears in the disambiguation
    dictionary.  Returns a list of suggestion dicts (label, phrase,
    description) that the frontend can render as clickable chips.
    """
    keywords = extract_keywords(query)
    if not keywords or len(keywords) > 2:
        return []

    suggestions: List[Dict] = []
    for kw in keywords:
        if kw in _AMBIGUOUS_WORDS:
            suggestions.extend(_AMBIGUOUS_WORDS[kw])

    return suggestions


def _expanded_queries(query: str, facet: str = "auto") -> List[Tuple[str, str]]:
    """Returns list of (subquery, facet_id) to run."""
    base = (query or "").strip()
    if not base:
        return []

    if not _is_drill_family(base):
        return [(base, "auto")]

    facet = (facet or "auto").lower()
    if facet == "oil_gas":
        return [(f"{base} oil and gas well rig offshore", "oil_gas")]
    if facet == "tools":
        return [(f"{base} drilling tools drill bit torque motor", "tools")]
    if facet == "analytics":
        return [(f"{base} drill down into data dashboard kpi", "analytics")]

    return [
        (base, "auto"),
        ("drilling tools", "tools"),
        ("oil and gas drilling", "oil_gas"),
        ("drill down into data", "analytics"),
    ]


def _diversified_merge(
    per_facet: Dict[str, List["SearchResult"]], top_k: int
) -> List["SearchResult"]:
    """Round-robin interleaving across facets, with dedup by (video_id, segment_id, start_time)."""
    if not per_facet:
        return []

    facet_lists = {
        k: sorted(v, key=lambda r: r.score, reverse=True) for k, v in per_facet.items()
    }
    facet_order = [
        k for k in ["auto", "oil_gas", "tools", "analytics"] if k in facet_lists
    ] + [
        k
        for k in facet_lists.keys()
        if k not in {"auto", "oil_gas", "tools", "analytics"}
    ]

    out: List["SearchResult"] = []
    seen = set()
    idx = {k: 0 for k in facet_lists.keys()}

    while len(out) < top_k:
        progressed = False
        for facet_id in facet_order:
            items = facet_lists.get(facet_id) or []
            i = idx[facet_id]
            if i >= len(items):
                continue
            r = items[i]
            idx[facet_id] += 1
            progressed = True

            key = (r.video_id, r.segment_id, round(float(r.start_time), 2))
            if key in seen:
                continue
            seen.add(key)

            r.facet = getattr(r, "facet", None) or facet_id
            out.append(r)
            if len(out) >= top_k:
                break
        if not progressed:
            break

    return out


@dataclass
class SearchResult:
    """Search result with video timestamp and text."""

    segment_id: int
    video_id: int
    video_filename: str
    video_path: str  # Full path to video file
    start_time: float
    end_time: float
    text: str
    score: float
    match_type: str  # "exact", "fuzzy", "semantic"
    result_id: Optional[int] = (
        None  # Optional ID from DB (transcript ID or negative scene ID)
    )
    keyframe_path: str = ""  # Path to keyframe image for thumbnails
    facet: Optional[str] = None  # Optional facet label: oil_gas/tools/analytics/auto
    evidence_time: Optional[float] = None  # Best frame/sample timestamp evidence
    evidence_frame_role: Optional[str] = None  # start/mid/end/extra_n
    source_type: str = "video"  # "video" or "document"
    document_page: Optional[int] = None  # 1-indexed page for document hits
    document_chunk_index: Optional[int] = None  # 0-indexed chunk order
    document_section_heading: Optional[str] = None
    document_file_type: Optional[str] = None
    result_language: Optional[str] = None  # ISO-ish language code when known

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        timestamp = f"{int(self.start_time // 3600):02d}:{int((self.start_time % 3600) // 60):02d}:{int(self.start_time % 60):02d}"
        document_location = None
        if self.source_type == "document":
            if self.document_page:
                document_location = f"Page {self.document_page}"
            elif self.document_chunk_index is not None:
                document_location = f"Chunk {int(self.document_chunk_index) + 1}"
            if document_location:
                timestamp = document_location

        payload = {
            "source_type": self.source_type,
            "segment_id": self.segment_id,
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "video_path": self.video_path,
            "timestamp": timestamp,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "text": self.text,
            "score": round(self.score, 4),
            "match_type": self.match_type,
            "result_id": self.result_id,
            "keyframe_path": self.keyframe_path,
            "facet": self.facet,
            "evidence_time": round(self.evidence_time, 2)
            if self.evidence_time is not None
            else None,
            "evidence_frame_role": self.evidence_frame_role,
            "document_id": self.video_id if self.source_type == "document" else None,
            "document_filename": self.video_filename
            if self.source_type == "document"
            else None,
            "document_path": self.video_path if self.source_type == "document" else None,
            "document_page": self.document_page,
            "document_chunk_index": self.document_chunk_index,
            "document_section_heading": self.document_section_heading,
            "document_file_type": self.document_file_type,
            "document_location": document_location,
            "language": self.result_language,
        }
        for extra_key in ("text_score", "vision_score", "combined_score"):
            val = getattr(self, extra_key, None)
            if val is not None:
                try:
                    payload[extra_key] = round(float(val), 4)
                except (TypeError, ValueError):
                    payload[extra_key] = val
        return payload


class SemanticSearchEngine:
    """Search engine with typo tolerance and semantic understanding."""

    def __init__(
        self,
        db: Session,
        # Caching options
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 3600,
        max_cache_size: int = 1000,
        # Parallel execution
        parallel_enabled: bool = True,
        # Reranker
        reranker_enabled: bool = False,
    ):
        """
        Initialize search engine.

        Args:
            db: Database session
            cache_enabled: Enable query result caching (OPTIMIZATION #5)
            cache_ttl_seconds: Cache time-to-live in seconds (default: 1 hour)
            max_cache_size: Maximum entries in memory cache
            parallel_enabled: Run semantic + fuzzy searches in parallel (OPTIMIZATION #4)
            reranker_enabled: Use cross-encoder reranker for improved precision
        """
        self.db = db
        self.embedding_gen = get_embedding_generator()

        # Caching configuration
        self.cache_enabled = cache_enabled
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.max_cache_size = max_cache_size
        self._memory_cache: Dict[str, Tuple[List[SearchResult], datetime]] = {}

        # Parallel execution
        self.parallel_enabled = parallel_enabled
        self._executor = ThreadPoolExecutor(max_workers=2) if parallel_enabled else None

        # Cross-encoder reranker (lazy loaded on first use)
        self.reranker_enabled = reranker_enabled
        self._reranker = None  # Loaded lazily

        # Performance statistics
        self.stats = {
            "queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_hits": 0,
            "db_hits": 0,
            "avg_latency_ms": 0.0,
        }
        self.query_translation_enabled = os.getenv(
            "SEARCH_QUERY_TRANSLATION_ENABLED", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.query_translation_timeout = float(
            os.getenv("SEARCH_QUERY_TRANSLATION_TIMEOUT", "1.5")
        )
        self.query_translation_targets = tuple(
            code.strip().lower()
            for code in os.getenv("SEARCH_QUERY_TRANSLATION_TARGETS", "en,no").split(",")
            if code.strip()
        )
        self._query_translation_cache: Dict[Tuple[str, str, str], Optional[str]] = {}

    @staticmethod
    def _normalize_lang_code(lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None
        code = str(lang).strip().lower()
        if not code:
            return None
        if code.startswith("no") or code in {"nb", "nn", "nor"}:
            return "no"
        if code.startswith("en"):
            return "en"
        return code

    def _detect_query_language(self, query: str) -> str:
        return "no" if self._is_norwegian_query(query) else "en"

    def _translate_query(
        self, query: str, source_lang: str, target_lang: str
    ) -> Optional[str]:
        source = self._normalize_lang_code(source_lang)
        target = self._normalize_lang_code(target_lang)
        if not query or not source or not target or source == target:
            return None

        cache_key = (query.strip().lower(), source, target)
        if cache_key in self._query_translation_cache:
            return self._query_translation_cache[cache_key]

        translated: Optional[str] = None
        try:
            lang_pair = f"{source}|{target}"
            url = (
                "https://api.mymemory.translated.net/get?"
                f"q={urllib.parse.quote(query[:500])}&langpair={lang_pair}"
            )
            with urllib.request.urlopen(url, timeout=self.query_translation_timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            candidate = (
                payload.get("responseData", {}).get("translatedText", "").strip()
            )
            if candidate and candidate.lower() != query.strip().lower():
                translated = candidate
        except Exception:
            translated = None

        self._query_translation_cache[cache_key] = translated
        return translated

    def _build_multilingual_query_variants(self, query: str) -> List[str]:
        base = (query or "").strip()
        if not base:
            return []

        variants: List[str] = [base]
        seen = {base.lower()}
        source_lang = self._detect_query_language(base)

        if self.query_translation_enabled:
            for target in self.query_translation_targets:
                target = self._normalize_lang_code(target)
                if not target or target == source_lang:
                    continue
                translated = self._translate_query(base, source_lang, target)
                if not translated:
                    continue
                key = translated.lower()
                if key not in seen:
                    variants.append(translated)
                    seen.add(key)

        return variants

    @staticmethod
    def _collect_anchor_keywords(queries: List[str]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for q in queries:
            for kw in extract_keywords(q):
                if kw not in seen:
                    seen.add(kw)
                    ordered.append(kw)
        return ordered

    def _fuzzy_text_search_multi(
        self,
        queries: List[str],
        top_k: int = 20,
        video_filter: Optional[str] = None,
    ) -> Dict:
        merged: Dict = {}
        for q in queries:
            partial = self._fuzzy_text_search(q, top_k=top_k, video_filter=video_filter)
            for key, entry in partial.items():
                existing = merged.get(key)
                if existing is None or entry[0] > existing[0]:
                    merged[key] = entry
        return merged

    def _apply_reranking(
        self,
        query: str,
        results: List["SearchResult"],
        top_k: int,
        deep_search: bool = False,
    ) -> List["SearchResult"]:
        """Apply reranking only when explicitly requested or globally enabled."""
        if not results or len(results) <= 1:
            return results

        if not (deep_search or self.reranker_enabled):
            return results

        if self._reranker is None:
            self._reranker = get_reranker(enabled=True)
        if not self._reranker:
            return results

        rerank_count = min(len(results), max(top_k, 1) * 3)
        reranked_head = self._reranker.rerank(
            query, results[:rerank_count], top_k=rerank_count
        )
        reranked_results = list(reranked_head) + list(results[rerank_count:])
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results

    def _build_vocabulary(self) -> Set[str]:
        """Build vocabulary from transcript segments for 'did you mean?' suggestions."""
        global _vocabulary
        if _vocabulary is not None:
            return _vocabulary

        try:
            result = self.db.execute(
                text("""
                SELECT DISTINCT unnest(string_to_array(lower(text), ' ')) AS word
                FROM transcript_segments
            """)
            )
            # Only keep words >= 3 chars, strip punctuation
            raw_words = {row[0].strip(".,!?'\"()-:;") for row in result}
            _vocabulary = {w for w in raw_words if len(w) >= 3 and w.isalpha()}
            print(f"Vocabulary built: {len(_vocabulary)} unique words")
        except Exception as e:
            print(f"Failed to build vocabulary: {e}")
            _vocabulary = set()

        return _vocabulary

    def _suggest_correction(self, query: str) -> Optional[str]:
        """
        Suggest a correction for the query using vocabulary matching.
        Only called when search returns no good results.

        Returns:
            Suggested corrected query, or None if no suggestion.
        """
        vocab = self._build_vocabulary()
        if not vocab:
            return None

        query_words = query.lower().split()
        corrected_words = []
        any_correction = False

        for word in query_words:
            clean_word = word.strip(".,!?'\"()-:;?")
            if len(clean_word) < 3 or clean_word in vocab:
                corrected_words.append(word)
                continue

            # Find close matches in vocabulary
            matches = get_close_matches(clean_word, vocab, n=1, cutoff=0.75)
            if matches:
                corrected_words.append(matches[0])
                any_correction = True
            else:
                corrected_words.append(word)

        if any_correction:
            return " ".join(corrected_words)
        return None

    def _fuzzy_match_score(self, query_term: str, text: str) -> float:
        """
        Calculate fuzzy matching score for a term in text.

        Args:
            query_term: Search term
            text: Text to search in

        Returns:
            Fuzzy match score (0-1)
        """
        query_term = query_term.lower()
        text = text.lower()

        # Exact match
        if query_term in text:
            return 1.0

        # Fuzzy match using sequence matcher
        best_score = 0.0
        words = text.split()

        for word in words:
            # Skip if length difference is too big (e.g. "oh" vs "omega")
            if abs(len(word) - len(query_term)) > 3:
                continue

            # Skip very short words for fuzzy matching (less than 3 chars) unless exact
            if len(query_term) < 3 and word != query_term:
                continue

            score = SequenceMatcher(None, query_term, word).ratio()

            # Boost score if one is substring of another
            if query_term in word or word in query_term:
                score = max(score, 0.9)

            best_score = max(best_score, score)

        return best_score

    def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.65,
        text_weight: float = 0.35,
        min_score: float = 0.15,
        video_filter: Optional[str] = None,
        log_query: bool = True,
        use_cache: bool = True,
        deep_search: bool = False,
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic similarity and fuzzy text matching.
        NOW OPTIMIZED with parallel execution (#4) and caching (#5)!

        Args:
            query: Search query (e.g., "where Omega Alpha well is discussed")
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity (0-1)
            text_weight: Weight for fuzzy text matching (0-1)
            min_score: Minimum combined score threshold
            video_filter: Optional video filename to filter results
            log_query: Log query to database for analytics
            use_cache: Use cached results if available (OPTIMIZATION #5)
            deep_search: If True, uses the expensive Cross-Encoder reranker

        Returns:
            List of SearchResult objects sorted by score
        """
        start_time = time.time()
        self.stats["queries"] += 1

        # OPTIMIZATION #5: Check cache if enabled
        cache_key = self._cache_key(
            query,
            top_k,
            semantic_weight=semantic_weight,
            text_weight=text_weight,
            min_score=min_score,
            video_filter=video_filter,
            deep_search=deep_search,
        )

        if self.cache_enabled and use_cache:
            # Memory cache first (fastest ~1ms)
            cached = self._check_memory_cache(cache_key)
            if cached:
                self._update_latency_stats((time.time() - start_time) * 1000)
                return cached[:top_k]

            # Database cache second (~5ms)
            cached = self._check_db_cache(cache_key)
            if cached:
                self._update_latency_stats((time.time() - start_time) * 1000)
                return cached[:top_k]

        self.stats["cache_misses"] += 1

        # Generate query embedding (semantic-first, no autocorrection)
        query_instruction = (
            "Given a query, retrieve relevant passages that answer the query\nQuery: "
        )
        query_embedding = self.embedding_gen.encode_single(
            query, instruction=query_instruction
        )

        query_variants = self._build_multilingual_query_variants(query)
        fuzzy_queries: List[str] = []
        for variant in query_variants:
            variant_keywords = extract_keywords(variant)
            if not variant_keywords:
                continue
            fuzzy_q = " ".join(variant_keywords)
            if fuzzy_q not in fuzzy_queries:
                fuzzy_queries.append(fuzzy_q)
        if not fuzzy_queries:
            fallback_keywords = extract_keywords(query)
            fallback_query = " ".join(fallback_keywords) if fallback_keywords else query
            fuzzy_queries = [fallback_query]

        # OPTIMIZATION #4: Parallel execution if enabled
        if self.parallel_enabled and self._executor:
            semantic_future = self._executor.submit(
                self._semantic_search,
                query_embedding,
                top_k=top_k * 3,
                video_filter=video_filter,
            )
            fuzzy_future = self._executor.submit(
                self._fuzzy_text_search_multi,
                fuzzy_queries,
                top_k=top_k * 3,
                video_filter=video_filter,
            )
            doc_future = self._executor.submit(
                self._document_semantic_search,
                query_embedding,
                top_k=top_k,
            )
            semantic_results = semantic_future.result()
            fuzzy_results = fuzzy_future.result()
            doc_results = doc_future.result()
        else:
            semantic_results = self._semantic_search(
                query_embedding, top_k=top_k * 3, video_filter=video_filter
            )
            fuzzy_results = self._fuzzy_text_search_multi(
                fuzzy_queries, top_k=top_k * 3, video_filter=video_filter
            )
            doc_results = self._document_semantic_search(
                query_embedding, top_k=top_k
            )

        # Combine and re-rank results
        combined_results = self._combine_results(
            query,
            semantic_results,
            fuzzy_results,
            semantic_weight=semantic_weight,
            text_weight=text_weight,
            anchor_queries=query_variants,
        )

        # Merge document results into combined results
        if doc_results:
            combined_results.extend(doc_results)

        # Filter out extremely short noisy segments (e.g. "Talking", "Except")
        combined_results = [r for r in combined_results if len(r.text.strip()) > 7]

        # Sort by score first
        combined_results.sort(key=lambda x: x.score, reverse=True)

        # ── Cross-encoder reranking (explicit slow path) ──
        combined_results = self._apply_reranking(
            query,
            combined_results,
            top_k=top_k,
            deep_search=deep_search,
        )

        # ── Keyword anchoring guardrail (prevents semantic overreach) ──
        # Apply AFTER reranking so the reranker cannot re-introduce unanchored hits.
        # Uses Norwegian stem matching so "brønnstrømmer" anchors on "brønnstrømmen".
        keywords = self._collect_anchor_keywords(query_variants)
        if keywords and not _is_analytics_intent(query):
            short_query = (len(keywords) <= 2) or (len(query.strip()) <= 14)
            query_lang = self._detect_query_language(query)
            anchored = []
            unanchored = []
            for r in combined_results:
                result_lang = self._normalize_lang_code(
                    getattr(r, "result_language", None)
                )
                is_cross_language = bool(
                    result_lang and query_lang and result_lang != query_lang
                )
                if is_cross_language:
                    has_anchor = True
                else:
                    has_anchor = any(
                        _whole_word_in_text(k, r.text) or stem_matches_in_text(k, r.text)
                        for k in keywords
                    )
                if has_anchor:
                    anchored.append(r)
                else:
                    unanchored.append(r)

            penalty = 0.35 if short_query else 0.65
            for r in unanchored:
                r.score *= penalty

            combined_results = anchored + unanchored
            combined_results.sort(key=lambda x: x.score, reverse=True)

        # Filter by minimum score
        combined_results = [r for r in combined_results if r.score >= min_score]

        # Dynamic/Relative Filtering: Drop results that are much worse than top result
        if combined_results:
            top_score = combined_results[0].score
            # Keep results within 60% of top score (was 75%, relaxed for better recall)
            relative_threshold = top_score * 0.60
            combined_results = [
                r for r in combined_results if r.score >= relative_threshold
            ]

        # Deduplicate overlapping segments from the same video (keep higher scored)
        combined_results = self._deduplicate_overlapping(
            combined_results, overlap_window=5.0
        )

        final_results = combined_results[:top_k]

        # OPTIMIZATION #5: Save to cache
        if self.cache_enabled and final_results:
            cache_params = {
                "top_k": top_k,
                "semantic_weight": semantic_weight,
                "text_weight": text_weight,
                "min_score": min_score,
                "video_filter": video_filter,
            }
            self._save_to_cache(cache_key, query, final_results, cache_params)

        # Log query
        if log_query and final_results:
            self._log_query(
                query,
                query_embedding.tolist(),
                len(final_results),
                final_results[0].segment_id,
            )

        self._update_latency_stats((time.time() - start_time) * 1000)
        return final_results

    def search_with_fallback(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
        log_query: bool = True,
        facet: str = "auto",
        deep_search: bool = False,
    ) -> Dict:
        """
        Tiered search with fallback strategy (Google-like behavior).

        Tier 1: Full query search (semantic + fuzzy)
        Tier 2: Relaxed thresholds if Tier 1 returns too few results
        Tier 3: Word decomposition — search individual words, merge results

        Returns:
            Dict with 'results', 'search_metadata' containing strategy info
        """
        # Semantic-first: use the original query directly (no autocorrection)
        search_query = query

        metadata = {
            "original_query": query,
            "corrected_query": None,
            "corrections": [],
            "did_you_mean": None,
            "search_strategy": "exact",
            "search_message": None,
            "tiers_tried": [],
            "keywords_used": None,
            "facet_applied": (facet or "auto"),
            "facets": _facet_suggestions_for_query(query),
            "sense_suggestions": _sense_suggestions(query),
        }

        def finalize_results(candidate_results: List[SearchResult]) -> List[SearchResult]:
            final = self._apply_reranking(
                search_query,
                list(candidate_results or []),
                top_k=top_k,
                deep_search=deep_search,
            )
            return final[:top_k]

        # ── Query preprocessing: extract content keywords ──
        keywords = extract_keywords(search_query)
        metadata["keywords_used"] = keywords

        # If NO meaningful keywords remain, the query is all stop/command words
        # (e.g., "show me all videos", "the", "tell me")
        if not keywords:
            metadata["search_strategy"] = "no_keywords"
            metadata["search_message"] = (
                f'Your query "{search_query}" contains only common words. '
                f'Try specific keywords like "drilling", "safety equipment", '
                f'"offshore platform", etc.'
            )
            return {"results": [], "search_metadata": metadata}

        # Avoid Unicode arrows for Windows cp1252 consoles.
        print(f'  Query: "{search_query}" -> Keywords: {keywords}')

        # ── Ambiguity handling: query expansion + diversified merge ──
        expanded = _expanded_queries(search_query, facet=facet or "auto")
        if len(expanded) > 1:
            metadata["tiers_tried"].append("expanded_queries")
            per_facet_results: Dict[str, List[SearchResult]] = {}
            for subq, facet_id in expanded:
                sub_results = self.search(
                    query=subq,
                    top_k=max(top_k, 8),
                    min_score=0.18,
                    video_filter=video_filter,
                    log_query=False,  # log only once
                    deep_search=False,
                )
                for r in sub_results:
                    r.facet = facet_id
                per_facet_results.setdefault(facet_id, []).extend(sub_results)

            results = _diversified_merge(per_facet_results, top_k=top_k)
            if results:
                metadata["search_strategy"] = "diversified"
                metadata["search_message"] = (
                    "Showing a balanced mix across meanings. Use the chips to focus."
                )
                return {
                    "results": finalize_results(results),
                    "search_metadata": metadata,
                }

        # ── Tier 1: Full query search with standard thresholds ──
        results = self.search(
            query=search_query,
            top_k=top_k,
            min_score=0.20,
            video_filter=video_filter,
            log_query=log_query,
            deep_search=False,
        )
        metadata["tiers_tried"].append("full_query")

        # Check quality: enough results with decent scores?
        good_results = [r for r in results if r.score >= 0.30]
        if len(good_results) >= min(3, top_k):
            metadata["search_strategy"] = "direct"
            return {"results": results, "search_metadata": metadata}

        # ── Tier 2: Relaxed thresholds ──
        relaxed_results = self.search(
            query=search_query,
            top_k=top_k,
            min_score=0.10,
            video_filter=video_filter,
            log_query=False,  # Don't double-log
            deep_search=False,
        )
        metadata["tiers_tried"].append("relaxed")

        if len(relaxed_results) > len(results):
            results = relaxed_results

        good_results = [r for r in results if r.score >= 0.20]
        if len(good_results) >= min(2, top_k):
            metadata["search_strategy"] = "relaxed"
            metadata["search_message"] = (
                f'Showing best available matches for "{search_query}"'
            )
            return {"results": results, "search_metadata": metadata}

        # ── Tier 3a: Keyword-phrase search ──────────────────────────────────
        # Before splitting into individual words, try the extracted keywords
        # joined as a phrase. This keeps named entities ("Deepsea Stavanger",
        # "Omega Alpha well") intact and avoids misleading "individual terms"
        # messages when a specific proper noun just isn't in transcripts.
        words = keywords  # already cleaned of stop words

        if len(words) > 1:
            phrase_query = " ".join(words)
            metadata["tiers_tried"].append("phrase")
            phrase_results = self.search(
                query=phrase_query,
                top_k=top_k,
                min_score=0.15,
                video_filter=video_filter,
                log_query=False,
                deep_search=False,
            )

            if phrase_results and phrase_results[0].score > (
                results[0].score if results else 0
            ):
                results = phrase_results

            # If phrase search produced enough good hits, stop here
            good_results = [r for r in results if r.score >= 0.20]
            if len(good_results) >= min(2, top_k):
                metadata["search_strategy"] = "phrase"
                metadata["search_message"] = (
                    f'Showing best matches for "{phrase_query}"'
                )
                return {"results": results, "search_metadata": metadata}

        # ── Tier 3b: Individual word decomposition (last resort) ─────────────
        # Only reached when Tier 1, 2, and 3a all fail to produce enough hits.
        if len(words) > 1:
            metadata["tiers_tried"].append("decomposed")
            all_word_results = {}
            active_facet = (facet or "auto").lower()
            facet_labels = {
                "auto": "All meanings",
                "oil_gas": "Oil & gas",
                "tools": "Tools",
                "analytics": "Data / drill-down",
            }

            for word in words:
                word_results = self.search(
                    query=word,
                    top_k=top_k,
                    min_score=0.30,
                    video_filter=video_filter,
                    log_query=False,
                    deep_search=False,
                )
                for r in word_results:
                    key = r.segment_id
                    if key in all_word_results:
                        existing = all_word_results[key]
                        existing.score = max(existing.score, r.score) * 1.10
                    else:
                        all_word_results[key] = r

            decomposed_results = sorted(
                all_word_results.values(), key=lambda x: x.score, reverse=True
            )[:top_k]

            if decomposed_results and (
                not results
                or decomposed_results[0].score > (results[0].score if results else 0)
            ):
                results = decomposed_results
                metadata["search_strategy"] = "expanded"
                if active_facet != "auto":
                    metadata["search_message"] = (
                        f'Showing related matches in "{facet_labels.get(active_facet, active_facet)}" '
                        f'for "{search_query}".'
                    )
                else:
                    metadata["search_message"] = (
                        f'Couldn\'t find "{search_query}" as a phrase. '
                        f"Showing similar results for related terms."
                    )
            elif results:
                metadata["search_strategy"] = "relaxed"
                metadata["search_message"] = (
                    f'Showing best available matches for "{search_query}"'
                )

        # ── "Did you mean?" suggestion (only when no/poor results) ──
        if not results or (results and results[0].score < 0.20):
            suggestion = self._suggest_correction(query)
            if suggestion and suggestion.lower() != query.lower():
                metadata["did_you_mean"] = suggestion
                metadata["search_message"] = (
                    f'No results found for "{query}". Did you mean "{suggestion}"?'
                )

        # Final fallback message if still no results
        if not results and not metadata.get("did_you_mean"):
            metadata["search_strategy"] = "no_results"
            metadata["search_message"] = (
                f'No results found for "{search_query}". '
                f"Try simpler or different keywords."
            )

        return {"results": finalize_results(results), "search_metadata": metadata}

    def _semantic_search(
        self, query_embedding, top_k: int = 20, video_filter: Optional[str] = None
    ) -> Dict[int, Tuple[float, TranscriptSegment]]:
        """
        Semantic search using vector similarity.

        Returns:
            Dict mapping segment_id -> (score, segment)
        """
        # Use pgvector cosine similarity (1 - cosine_distance)
        query_filter = ""
        if video_filter:
            query_filter = "AND v.filename = :video_filter"

        sql_query = text(f"""
            SELECT 
                ts.id as segment_id,
                v.id as video_id,
                v.filename,
                v.file_path,
                COALESCE(ts.start_time, s.start_time) as start_time,
                COALESCE(ts.end_time, s.end_time) as end_time,
                ts.text as transcript_text,
                ts.language as transcript_language,
                s.caption as scene_caption,
                s.object_labels as scene_object_labels,
                s.ocr_text as scene_ocr_text,
                1 - (e.embedding <=> CAST(:query_embedding AS vector)) AS similarity,
                s.scene_id,
                ve.keyframe_path
            FROM embeddings e
            LEFT JOIN transcript_segments ts ON e.segment_id = ts.id
            LEFT JOIN scenes s ON (e.scene_id = s.id OR ts.scene_id = s.id)
            LEFT JOIN LATERAL (
                SELECT ve1.keyframe_path
                FROM visual_embeddings ve1
                WHERE ve1.scene_id = s.id
                ORDER BY CASE ve1.frame_role
                    WHEN 'mid' THEN 0
                    WHEN 'start' THEN 1
                    WHEN 'end' THEN 2
                    ELSE 3
                END,
                COALESCE(ve1.sample_time, 0)
                LIMIT 1
            ) ve ON TRUE
            JOIN videos v ON (ts.video_id = v.id OR s.video_id = v.id)
            WHERE 1=1 {query_filter}
            ORDER BY e.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
        """)

        params = {"query_embedding": query_embedding.tolist(), "top_k": top_k}
        if video_filter:
            params["video_filter"] = video_filter

        results = self.db.execute(sql_query, params).fetchall()

        semantic_scores = {}
        for row in results:
            # Check column count and names from row
            segment_id = row.segment_id if row.segment_id else 0
            video_id = row.video_id
            filename = row.filename
            file_path = row.file_path
            start = row.start_time
            end = row.end_time
            if segment_id:
                segment_text = row.transcript_text or ""
            else:
                scene_parts = []
                scene_caption = str(row.scene_caption).strip() if row.scene_caption else ""
                if scene_caption.lower() in {"none", "null", "n/a", "na"}:
                    scene_caption = ""
                if scene_caption:
                    scene_parts.append(scene_caption)

                labels_text = ""
                if isinstance(row.scene_object_labels, list):
                    labels_text = " ".join(
                        str(lbl).strip()
                        for lbl in row.scene_object_labels
                        if str(lbl).strip()
                    )
                elif row.scene_object_labels:
                    labels_text = (
                        str(row.scene_object_labels)
                        .replace("[", " ")
                        .replace("]", " ")
                        .replace('"', " ")
                        .replace(",", " ")
                    )
                    labels_text = re.sub(r"\s+", " ", labels_text).strip()

                if labels_text:
                    scene_parts.append(labels_text)

                scene_ocr = str(row.scene_ocr_text).strip() if row.scene_ocr_text else ""
                if scene_ocr.lower() in {"none", "null", "n/a", "na"}:
                    scene_ocr = ""
                if scene_ocr:
                    scene_parts.append(f"[OCR] {scene_ocr}")

                scene_text = " ".join(scene_parts).strip()
                segment_text = (
                    f"[Visual] {scene_text}" if scene_text else "[Scene match: visual]"
                )
            similarity = row.similarity
            scene_val = row.scene_id
            keyframe_path = row.keyframe_path

            # Create a pseudo-segment or real segment
            segment = TranscriptSegment(
                id=segment_id,
                video_id=video_id,
                start_time=start,
                end_time=end,
                text=segment_text
                if segment_id
                else f"{segment_text} (Scene {scene_val})",
            )
            # Store video filename and path in custom attributes
            segment.video_filename = filename
            segment.video_path = file_path
            segment.language = self._normalize_lang_code(
                getattr(row, "transcript_language", None)
            )

            # Use segment_id if available, otherwise a unique key based on scene
            key = segment_id if segment_id else f"scene_{scene_val}_{video_id}"
            semantic_scores[key] = (similarity, segment, keyframe_path)

        return semantic_scores

    def _document_semantic_search(
        self, query_embedding, top_k: int = 10
    ) -> List["SearchResult"]:
        """
        Search document embeddings for matching chunks.
        Returns a list of SearchResult objects with source_type='document'.
        """
        try:
            sql_query = text("""
                SELECT
                    dc.id AS chunk_id,
                    d.id AS document_id,
                    d.filename,
                    d.file_path,
                    d.file_type,
                    d.language,
                    dc.chunk_index,
                    dc.page_number,
                    dc.section_heading,
                    dc.text,
                    dc.summary,
                    1 - (de.embedding <=> CAST(:query_embedding AS vector)) AS similarity
                FROM document_embeddings de
                JOIN document_chunks dc ON de.chunk_id = dc.id
                JOIN documents d ON dc.document_id = d.id
                ORDER BY de.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :top_k
            """)

            params = {"query_embedding": query_embedding.tolist(), "top_k": top_k}
            rows = self.db.execute(sql_query, params).fetchall()
        except Exception as e:
            # Table may not exist yet — fail silently
            return []

        results = []
        for row in rows:
            display_text = row.text or ""
            if row.summary:
                display_text = f"{row.summary}\n{display_text}"
            if row.section_heading:
                display_text = f"[{row.section_heading}] {display_text}"
            display_text = f"[Document: {row.filename}" + (
                f", p.{row.page_number}" if row.page_number else ""
            ) + f"] {display_text}"

            sr = SearchResult(
                segment_id=row.chunk_id,
                video_id=row.document_id,  # repurpose field for doc ID
                video_filename=row.filename,
                video_path=row.file_path,
                start_time=0.0,
                end_time=0.0,
                text=display_text,
                score=float(row.similarity),
                match_type="semantic",
                result_id=row.chunk_id,
                source_type="document",
                document_page=row.page_number,
                document_chunk_index=row.chunk_index,
                document_section_heading=row.section_heading,
                document_file_type=row.file_type,
                result_language=self._normalize_lang_code(row.language),
            )
            results.append(sr)

        return results

    def _is_norwegian_query(self, query: str) -> bool:
        """Heuristic: return True if the query looks Norwegian."""
        norwegian_markers = {
            'hva', 'er', 'og', 'på', 'som', 'til', 'fra', 'med',
            'det', 'den', 'de', 'en', 'et', 'av', 'for', 'om',
            'kan', 'har', 'var', 'vil', 'skal', 'hvor', 'når',
            'ikke', 'eller', 'men', 'også', 'alle', 'denne',
        }
        words = set(re.findall(r'[\w]+', query.lower(), flags=re.UNICODE))
        # Norwegian chars are a strong signal.
        has_no_chars = bool(re.search(r'[æøå]', query.lower()))
        marker_count = len(words & norwegian_markers)
        return has_no_chars or marker_count >= 2

    def _fuzzy_text_search(
        self, query: str, top_k: int = 20, video_filter: Optional[str] = None
    ) -> Dict[int, Tuple[float, TranscriptSegment]]:
        """
        Fuzzy text search using PostgreSQL full-text search.

        Uses a UNION of two query paths:
        1. Transcript text matches (standard path)
        2. Direct OCR scene matches (searches scenes.ocr_text directly)

        This avoids the problem where transcript_segments.scene_id is NULL
        for most segments, making OCR text unreachable via JOINs.

        Returns:
            Dict mapping segment_id -> (score, segment)
        """
        # PostgreSQL full-text search
        query_filter_ts = ""
        query_filter_ocr = ""
        if video_filter:
            query_filter_ts = "AND v.filename = :video_filter"
            query_filter_ocr = "AND v.filename = :video_filter"

        # Pre-filter: strip stop words from fuzzy query to avoid noise matches
        fuzzy_keywords = extract_keywords(query)
        clean_query = " ".join(fuzzy_keywords) if fuzzy_keywords else query

        # If all words are stop words, skip fuzzy search entirely
        if not fuzzy_keywords:
            return {}

        # Use Norwegian text-search config when query looks Norwegian
        # ('norwegian' does proper stemming; 'simple' does none).
        ts_cfg = "'norwegian'" if self._is_norwegian_query(query) else "'simple'"

        # UNION query: transcript matches + direct OCR scene matches
        sql_query = text(f"""
            WITH combined AS (
                -- Branch 1: Transcript text matches
                SELECT 
                    ts.id AS result_id,
                    ts.video_id,
                    v.filename,
                    v.file_path,
                    ts.start_time,
                    ts.end_time,
                    ts.text AS result_text,
                    ts_rank(to_tsvector({ts_cfg}, ts.text), websearch_to_tsquery({ts_cfg}, :query)) AS rank,
                    NULL AS ocr_text,
                    ve.keyframe_path,
                    ts.language AS result_language,
                    'transcript' AS match_source
                FROM transcript_segments ts
                JOIN videos v ON ts.video_id = v.id
                LEFT JOIN scenes s ON ts.scene_id = s.id
                LEFT JOIN LATERAL (
                    SELECT ve1.keyframe_path
                    FROM visual_embeddings ve1
                    WHERE ve1.scene_id = s.id
                    ORDER BY CASE ve1.frame_role
                        WHEN 'mid' THEN 0
                        WHEN 'start' THEN 1
                        WHEN 'end' THEN 2
                        ELSE 3
                    END,
                    COALESCE(ve1.sample_time, 0)
                    LIMIT 1
                ) ve ON TRUE
                WHERE to_tsvector({ts_cfg}, ts.text) @@ websearch_to_tsquery({ts_cfg}, :query)
                {query_filter_ts}

                UNION ALL

                -- Branch 2: Direct OCR scene matches (searches scenes.ocr_text directly)
                SELECT 
                    -(s.id) AS result_id,
                    s.video_id,
                    v.filename,
                    v.file_path,
                    s.start_time,
                    s.end_time,
                    '[OCR] ' || s.ocr_text AS result_text,
                                        ts_rank(
                                                to_tsvector({ts_cfg}, COALESCE(s.ocr_text_norm, s.ocr_text)),
                                                websearch_to_tsquery({ts_cfg}, :query)
                                        ) * (0.8 + 0.2 * COALESCE(s.ocr_confidence, 0.6)) AS rank,
                    s.ocr_text,
                    COALESCE(ve.keyframe_path, s.keyframe_path) AS keyframe_path,
                    NULL::text AS result_language,
                    'ocr' AS match_source
                FROM scenes s
                JOIN videos v ON s.video_id = v.id
                                LEFT JOIN LATERAL (
                                        SELECT ve1.keyframe_path
                                        FROM visual_embeddings ve1
                                        WHERE ve1.scene_id = s.id
                                        ORDER BY CASE ve1.frame_role
                                                WHEN 'mid' THEN 0
                                                WHEN 'start' THEN 1
                                                WHEN 'end' THEN 2
                                                ELSE 3
                                        END,
                                        COALESCE(ve1.sample_time, 0)
                                        LIMIT 1
                                ) ve ON TRUE
                WHERE s.ocr_text IS NOT NULL
                                    AND to_tsvector({ts_cfg}, COALESCE(s.ocr_text_norm, s.ocr_text)) @@ websearch_to_tsquery({ts_cfg}, :query)
                {query_filter_ocr}

                UNION ALL

                -- Branch 3: Visual caption + object-label matches (Qwen2-VL enrichment)
                -- Uses negative offset -2000000 to avoid ID collision with OCR branch
                SELECT
                    -(s.id + 2000000) AS result_id,
                    s.video_id,
                    v.filename,
                    v.file_path,
                    s.start_time,
                    s.end_time,
                    '[Visual] ' || TRIM(BOTH ' ' FROM
                        CASE
                            WHEN s.caption IS NULL OR LOWER(BTRIM(s.caption)) IN ('none', 'null', 'n/a', 'na')
                            THEN ''
                            ELSE BTRIM(s.caption)
                        END || ' ' ||
                        COALESCE(regexp_replace(COALESCE(s.object_labels::text, '[]'), '[\\[\\]\"]', ' ', 'g'), '') || ' ' ||
                        CASE
                            WHEN s.ocr_text IS NULL OR LOWER(BTRIM(s.ocr_text)) IN ('none', 'null', 'n/a', 'na')
                            THEN ''
                            ELSE BTRIM(s.ocr_text)
                        END
                    ) AS result_text,
                    -- Boost Visual rank 6x so richer descriptions surface clearly
                    ts_rank(
                        to_tsvector({ts_cfg},
                            TRIM(BOTH ' ' FROM
                                CASE
                                    WHEN s.caption IS NULL OR LOWER(BTRIM(s.caption)) IN ('none', 'null', 'n/a', 'na')
                                    THEN ''
                                    ELSE BTRIM(s.caption)
                                END || ' ' ||
                                COALESCE(regexp_replace(COALESCE(s.object_labels::text, '[]'), '[\\[\\]\"]', ' ', 'g'), '') || ' ' ||
                                CASE
                                    WHEN s.ocr_text IS NULL OR LOWER(BTRIM(s.ocr_text)) IN ('none', 'null', 'n/a', 'na')
                                    THEN ''
                                    ELSE BTRIM(s.ocr_text)
                                END
                            )
                        ),
                        websearch_to_tsquery({ts_cfg}, :query)
                    ) * 6.0 AS rank,
                    NULL AS ocr_text,
                    COALESCE(ve.keyframe_path, s.keyframe_path) AS keyframe_path,
                    NULL::text AS result_language,
                    'visual' AS match_source
                FROM scenes s
                JOIN videos v ON s.video_id = v.id
                                LEFT JOIN LATERAL (
                                        SELECT ve1.keyframe_path
                                        FROM visual_embeddings ve1
                                        WHERE ve1.scene_id = s.id
                                        ORDER BY CASE ve1.frame_role
                                                WHEN 'mid' THEN 0
                                                WHEN 'start' THEN 1
                                                WHEN 'end' THEN 2
                                                ELSE 3
                                        END,
                                        COALESCE(ve1.sample_time, 0)
                                        LIMIT 1
                                ) ve ON TRUE
                WHERE (
                        (
                            s.caption IS NOT NULL
                            AND BTRIM(s.caption) <> ''
                            AND LOWER(BTRIM(s.caption)) NOT IN ('none', 'null', 'n/a', 'na')
                        )
                        OR (s.object_labels IS NOT NULL AND s.object_labels::text <> '[]')
                        OR (
                            s.ocr_text IS NOT NULL
                            AND BTRIM(s.ocr_text) <> ''
                            AND LOWER(BTRIM(s.ocr_text)) NOT IN ('none', 'null', 'n/a', 'na')
                        )
                      )
                  AND to_tsvector({ts_cfg},
                        TRIM(BOTH ' ' FROM
                            CASE
                                WHEN s.caption IS NULL OR LOWER(BTRIM(s.caption)) IN ('none', 'null', 'n/a', 'na')
                                THEN ''
                                ELSE BTRIM(s.caption)
                            END || ' ' ||
                            COALESCE(regexp_replace(COALESCE(s.object_labels::text, '[]'), '[\\[\\]\"]', ' ', 'g'), '') || ' ' ||
                            CASE
                                WHEN s.ocr_text IS NULL OR LOWER(BTRIM(s.ocr_text)) IN ('none', 'null', 'n/a', 'na')
                                THEN ''
                                ELSE BTRIM(s.ocr_text)
                            END
                        )
                      ) @@ websearch_to_tsquery({ts_cfg}, :query)
                {query_filter_ocr}
            )
            SELECT * FROM combined
            ORDER BY rank DESC
            LIMIT :top_k
        """)

        params = {"query": clean_query, "top_k": top_k}
        if video_filter:
            params["video_filter"] = video_filter

        results = self.db.execute(sql_query, params).fetchall()

        fuzzy_scores = {}
        for row in results:
            result_id = row.result_id
            video_id = row.video_id
            filename = row.filename
            file_path = row.file_path
            start = row.start_time
            end = row.end_time
            result_text = row.result_text
            rank = row.rank
            keyframe_path = row.keyframe_path
            result_language = row.result_language
            match_source = row.match_source

            segment = TranscriptSegment(
                id=abs(result_id),
                video_id=video_id,
                start_time=start,
                end_time=end,
                text=result_text,
            )
            segment.video_filename = filename
            segment.video_path = file_path
            segment.language = self._normalize_lang_code(result_language)

            # Use a unique key to avoid collisions between transcript, OCR, and visual results
            if match_source == "ocr":
                key = f"ocr_{abs(result_id)}"
            elif match_source == "visual":
                key = f"visual_{abs(result_id)}"
            else:
                key = result_id
            fuzzy_scores[key] = (rank, segment, keyframe_path)

        return fuzzy_scores

    def _deduplicate_overlapping(
        self,
        results: List[SearchResult],
        overlap_window: float = 5.0,
    ) -> List[SearchResult]:
        """Remove near-duplicate results from the same video within a time window."""
        if not results:
            return results

        deduplicated = []
        for result in results:
            # Document hits do not carry timeline offsets, so overlap-based
            # video deduplication is not meaningful for them.
            if getattr(result, "source_type", "video") == "document":
                deduplicated.append(result)
                continue
            is_duplicate = False
            for kept in deduplicated:
                if getattr(kept, "source_type", "video") == "document":
                    continue
                if (
                    result.video_id == kept.video_id
                    and abs(result.start_time - kept.start_time) < overlap_window
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(result)

        return deduplicated

    def _exact_keyword_boost(self, keywords: List[str], precompiled_regexes: Dict[str, re.Pattern], text: str) -> float:
        """Compute boost for exact/stem keyword presence in text.

        Returns a value in [0, 1] representing the fraction of query
        keywords that appear (literally or via Norwegian stemming) in
        the document text.
        """
        if not keywords:
            return 0.0

        text_lower = text.lower()
        matches = 0
        for kw in keywords:
            # Exact whole-word match using precompiled regex
            pattern = precompiled_regexes.get(kw)
            if pattern and pattern.search(text_lower):
                matches += 1
                continue
            
            # Norwegian stem-based match
            if stem_matches_in_text(kw, text_lower):
                matches += 1

        return matches / len(keywords)

    def _combine_results(
        self,
        query: str,
        semantic_results: Dict,
        fuzzy_results: Dict,
        semantic_weight: float = 0.7,
        text_weight: float = 0.3,
        anchor_queries: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Combine semantic and fuzzy search results with weighted scoring."""
        # Get all unique segment IDs
        all_segment_ids = set(semantic_results.keys()) | set(fuzzy_results.keys())

        # Find max fuzzy score for rank-based normalization
        max_fuzzy = 0.0
        for key in all_segment_ids:
            fuzzy_entry = fuzzy_results.get(key, (0, None, None))
            max_fuzzy = max(max_fuzzy, fuzzy_entry[0])

        # Pre-compute query keywords and regex patterns for exact boosting.
        # Include translated variants so lexical boosts work cross-language too.
        anchor_inputs = list(anchor_queries or [])
        if query not in anchor_inputs:
            anchor_inputs.append(query)
        query_keywords = self._collect_anchor_keywords(anchor_inputs)
        
        precompiled_regexes = {}
        for kw in query_keywords:
            try:
                # (?<![\w]) and (?![\w]) are Unicode-aware word boundaries
                precompiled_regexes[kw] = re.compile(rf"(?<![\w]){re.escape(kw.lower())}(?![\w])", flags=re.UNICODE)
            except re.error:
                precompiled_regexes[kw] = re.compile(re.escape(kw.lower()))

        combined = []
        for key in all_segment_ids:
            # Get scores and keyframe path
            semantic_entry = semantic_results.get(key, (0, None, None))
            fuzzy_entry = fuzzy_results.get(key, (0, None, None))

            semantic_score = semantic_entry[0]
            fuzzy_score = fuzzy_entry[0]

            # Get segment
            segment = semantic_entry[1]
            if segment is None:
                segment = fuzzy_entry[1]

            # Get keyframe (prefer visual match if available)
            keyframe_path = semantic_entry[2]
            if not keyframe_path:
                keyframe_path = fuzzy_entry[2]

            segment_id = segment.id if segment.id else 0

            # Normalize fuzzy score relative to best fuzzy match (rank-based)
            fuzzy_score_norm = (fuzzy_score / max_fuzzy) if max_fuzzy > 0 else 0.0

            # Exact-keyword boost: up to +0.3 when all keywords are in text.
            # Gated on semantic score to prevent boosting polysemous words 
            # (e.g. "well done" instead of "oil well")
            exact_boost_val = self._exact_keyword_boost(query_keywords, precompiled_regexes, segment.text)
            exact_boost = 0.0
            if exact_boost_val > 0:
                if semantic_score > 0.12 or len(query_keywords) > 1:
                    exact_boost = exact_boost_val * 0.3

            # Combined score
            combined_score = (
                semantic_weight * semantic_score + text_weight * fuzzy_score_norm
            ) + exact_boost

            is_ocr_only = isinstance(key, str) and key.startswith("ocr_")
            is_visual_only = isinstance(key, str) and key.startswith("visual_")
            is_scene_semantic = isinstance(key, str) and key.startswith("scene_")

            # OCR-only matches (no semantic embedding) get a floor score
            # since exact text matches from keyframes are inherently high quality
            if is_ocr_only and semantic_score == 0 and fuzzy_score_norm > 0:
                combined_score = max(combined_score, 0.60 * fuzzy_score_norm)

            # Visual-caption matches also get a floor — Qwen2-VL captions are
            # rich descriptions and deserve to surface even when no semantic
            # embedding exists for the scene yet.
            if is_visual_only and semantic_score == 0 and fuzzy_score_norm > 0:
                combined_score = max(combined_score, 0.60 * fuzzy_score_norm)

            # Component scores exposed to frontend badges/tabs.
            text_component = max(
                0.0,
                min(1.0, max(float(semantic_score or 0.0), float(fuzzy_score_norm or 0.0))),
            )
            visual_component = (
                max(0.0, min(1.0, fuzzy_score_norm))
                if (is_ocr_only or is_visual_only)
                else 0.0
            )
            if is_scene_semantic and semantic_score > 0:
                visual_component = max(
                    visual_component, max(0.0, min(1.0, float(semantic_score)))
                )

            # Determine match type
            if semantic_score > 0.7:
                match_type = "semantic"
            elif fuzzy_score_norm > 0.5:
                match_type = "fuzzy"
            else:
                match_type = "hybrid"

            result = SearchResult(
                segment_id=segment_id,
                video_id=segment.video_id,
                video_filename=segment.video_filename,
                video_path=segment.video_path,
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=segment.text,
                score=combined_score,
                match_type=match_type,
                keyframe_path=keyframe_path or "",
                result_id=key,  # Pass the unique key as result_id
                result_language=self._normalize_lang_code(
                    getattr(segment, "language", None)
                ),
            )
            result.text_score = text_component
            result.vision_score = visual_component
            result.combined_score = combined_score

            combined.append(result)

        return combined

    def _log_query(
        self,
        query_text: str,
        query_embedding: List,
        results_count: int,
        top_result_id: int,
    ):
        """Log search query for analytics."""
        try:
            # Ensure top_result_id actually refers to an existing transcript segment.
            # Some results (visual/OCR-only) may not have a valid transcript segment id.
            from database.models import TranscriptSegment  # local import to avoid cycles

            valid_top_id = top_result_id
            if top_result_id is not None:
                exists = (
                    self.db.query(TranscriptSegment)
                    .filter(TranscriptSegment.id == top_result_id)
                    .first()
                )
                if exists is None:
                    valid_top_id = None

            query_log = SearchQuery(
                query_text=query_text,
                query_embedding=query_embedding,
                results_count=results_count,
                top_result_id=valid_top_id,
            )
            self.db.add(query_log)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            # Temporary compatibility path: query analytics should not fail search
            # when model and table vector dimensions are in migration.
            try:
                query_log = SearchQuery(
                    query_text=query_text,
                    query_embedding=None,
                    results_count=results_count,
                    top_result_id=valid_top_id,
                )
                self.db.add(query_log)
                self.db.commit()
            except Exception as retry_e:
                print(f"Warning: Failed to log query: {retry_e}")
                self.db.rollback()

    def search_exact_phrase(
        self, phrase: str, video_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for exact phrase match.

        Args:
            phrase: Exact phrase to search for
            video_filter: Optional video filename filter

        Returns:
            List of SearchResult objects
        """
        query = self.db.query(TranscriptSegment, Video).join(
            Video, TranscriptSegment.video_id == Video.id
        )

        # Case-insensitive exact match
        query = query.filter(TranscriptSegment.text.ilike(f"%{phrase}%"))

        if video_filter:
            query = query.filter(Video.filename == video_filter)

        results = query.all()

        search_results = []
        for segment, video in results:
            result = SearchResult(
                segment_id=segment.id,
                video_id=segment.video_id,
                video_filename=video.filename,
                video_path=video.file_path,
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=segment.text,
                score=1.0,  # Exact match
                match_type="exact",
            )
            search_results.append(result)

        return search_results

    # OPTIMIZATION #5: Caching methods
    def _cache_key(self, query: str, top_k: int, **kwargs) -> str:
        """Generate cache key from query parameters."""
        cache_data = {"query": query.lower().strip(), "top_k": top_k, **kwargs}
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _check_memory_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Check in-memory cache (fastest)."""
        if cache_key in self._memory_cache:
            results, timestamp = self._memory_cache[cache_key]

            if datetime.now() - timestamp < self.cache_ttl:
                self.stats["cache_hits"] += 1
                self.stats["memory_hits"] += 1
                return results
            else:
                # Expired - remove
                del self._memory_cache[cache_key]

        return None

    def _check_db_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Check database cache (persistent)."""
        try:
            result = self.db.execute(
                text("""
                SELECT cached_results 
                FROM query_cache 
                WHERE query_hash = :cache_key 
                AND expires_at > NOW()
                LIMIT 1
            """),
                {"cache_key": cache_key},
            )

            row = result.fetchone()
            if row:
                # Update hit count
                self.db.execute(
                    text("SELECT update_cache_stats(:cache_key)"),
                    {"cache_key": cache_key},
                )
                self.db.commit()

                # Deserialize results
                cached_data = row[0]
                allowed = set(SearchResult.__dataclass_fields__.keys())
                results: List[SearchResult] = []
                for item in cached_data:
                    base_payload = {k: v for k, v in item.items() if k in allowed}
                    result_obj = SearchResult(**base_payload)
                    for extra_key in ("text_score", "vision_score", "combined_score"):
                        if extra_key in item:
                            setattr(result_obj, extra_key, item.get(extra_key))
                    results.append(result_obj)

                # Add to memory cache
                self._memory_cache[cache_key] = (results, datetime.now())

                self.stats["cache_hits"] += 1
                self.stats["db_hits"] += 1
                return results

        except Exception as e:
            # If cache table doesn't exist yet, silently skip
            pass

        return None

    def _save_to_cache(
        self, cache_key: str, query: str, results: List[SearchResult], params: Dict
    ):
        """Save results to both memory and database cache."""
        # Memory cache
        self._memory_cache[cache_key] = (results, datetime.now())

        # Enforce max size (LRU eviction)
        if len(self._memory_cache) > self.max_cache_size:
            sorted_cache = sorted(self._memory_cache.items(), key=lambda x: x[1][1])
            self._memory_cache = dict(sorted_cache[-self.max_cache_size :])

        # Database cache (if table exists)
        try:
            serialized = [r.to_dict() for r in results]

            self.db.execute(
                text("""
                INSERT INTO query_cache (
                    query_text, query_hash, query_params, cached_results, expires_at
                )
                VALUES (
                    :query_text, :query_hash, :query_params::jsonb, 
                    :cached_results::jsonb, NOW() + :ttl_interval::interval
                )
                ON CONFLICT (query_hash) DO UPDATE
                SET cached_results = EXCLUDED.cached_results,
                    hit_count = query_cache.hit_count + 1,
                    last_used = NOW(),
                    expires_at = NOW() + :ttl_interval::interval
            """),
                {
                    "query_text": query,
                    "query_hash": cache_key,
                    "query_params": json.dumps(params),
                    "cached_results": json.dumps(serialized),
                    "ttl_interval": f"{self.cache_ttl.total_seconds()} seconds",
                },
            )
            self.db.commit()

        except Exception as e:
            # If cache table doesn't exist, skip DB caching
            self.db.rollback()

    def _update_latency_stats(self, latency_ms: float):
        """Update average latency statistics."""
        n = self.stats["queries"]
        current_avg = self.stats["avg_latency_ms"]
        self.stats["avg_latency_ms"] = (
            (current_avg * (n - 1) + latency_ms) / n if n > 0 else latency_ms
        )

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        total = self.stats["queries"]
        hit_rate = self.stats["cache_hits"] / total if total > 0 else 0

        return {
            "total_queries": total,
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "memory_hits": self.stats["memory_hits"],
            "db_hits": self.stats["db_hits"],
            "avg_latency_ms": round(self.stats["avg_latency_ms"], 2),
            "memory_cache_size": len(self._memory_cache),
            "parallel_enabled": self.parallel_enabled,
            "cache_enabled": self.cache_enabled,
        }

    def clear_cache(self, memory_only: bool = False):
        """Clear cache (useful for testing)."""
        self._memory_cache.clear()

        if not memory_only:
            try:
                self.db.execute(text("TRUNCATE TABLE query_cache"))
                self.db.commit()
            except:
                pass

    def cleanup_expired_cache(self):
        """Remove expired entries from database cache."""
        try:
            result = self.db.execute(text("SELECT clean_query_cache()"))
            deleted = result.fetchone()[0]
            self.db.commit()
            return deleted
        except:
            return 0

    def __del__(self):
        """Cleanup thread pool."""
        executor = getattr(self, "_executor", None)
        if executor:
            executor.shutdown(wait=False)
