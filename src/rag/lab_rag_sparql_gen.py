"""
RAG with RDF/SPARQL and a Local Small LLM
Project: AI in Architecture Knowledge Base
Authors: Clarisse Ballon, Léa Hadj-said — ESILV A4/S8 2025-2026

Requirements:
    pip install rdflib requests
    ollama pull gemma3:1b
    ollama serve (in a separate terminal)

Usage:
    python lab_rag_sparql_gen.py
"""

import re
import os
from typing import List, Tuple
from rdflib import Graph
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TTL_FILE = os.path.join(SCRIPT_DIR, "kb_graph.ttl")

if not os.path.exists(TTL_FILE):
    print(f"Warning: {TTL_FILE} not found")
    print(f"Available .ttl files: {[f for f in os.listdir(SCRIPT_DIR) if f.endswith('.ttl')]}")

OLLAMA_URL = "http://localhost:11434/api/generate"
GEMMA_MODEL = "gemma3:1b"
MAX_PREDICATES = 80
MAX_CLASSES = 40
SAMPLE_TRIPLES = 20

PREDICATE_LABELS = {
    "http://www.wikidata.org/prop/direct/P31":  "instance of",
    "http://www.wikidata.org/prop/direct/P279": "subclass of",
    "http://www.wikidata.org/prop/direct/P361": "part of",
    "http://www.wikidata.org/prop/direct/P527": "has part",
    "http://www.wikidata.org/prop/direct/P166": "award received",
    "http://www.wikidata.org/prop/direct/P69":  "educated at",
    "http://www.wikidata.org/prop/direct/P108": "employer",
    "http://www.wikidata.org/prop/direct/P463": "member of",
    "http://www.wikidata.org/prop/direct/P17":  "country",
    "http://www.wikidata.org/prop/direct/P921": "main subject",
    "http://www.wikidata.org/prop/direct/P84":  "architect",
    "http://www.wikidata.org/prop/direct/P178": "developer",
    "http://www.wikidata.org/prop/direct/P277": "programmed in",
    "http://www.wikidata.org/prop/direct/P170": "creator",
    "http://www.wikidata.org/prop/direct/P737": "influenced by",
    "http://www.wikidata.org/prop/direct/P941": "inspired by",
    "http://www.wikidata.org/prop/direct/P800": "notable work",
    "http://www.wikidata.org/prop/direct/P452": "industry",
    "http://www.wikidata.org/prop/direct/P136": "genre",
    "http://www.wikidata.org/prop/direct/P155": "follows",
    "http://www.wikidata.org/prop/direct/P156": "followed by",
}

# Entity name → Wikidata QID mapping for keyword detection
ENTITY_MAP = {
    "microsoft": "Q2283",
    "google": "Q95",
    "apple": "Q312",
    "amazon": "Q3884",
    "autodesk": "Q334535",
    "archicad": "Q634811",
    "alan turing": "Q7251",
    "germany": "Q183",
    "france": "Q142",
    "united states": "Q30",
    "china": "Q148",
}


# ----------------------------
# 0) Utility: call local LLM
# ----------------------------
def ask_local_llm(prompt: str, model: str = GEMMA_MODEL) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")
        return response.json().get("response", "")
    except Exception as e:
        return f"[LLM Error: {e}]"


# ----------------------------
# 1) Load RDF graph
# ----------------------------
def load_graph(ttl_path: str) -> Graph:
    g = Graph()
    g.parse(ttl_path, format="turtle")
    print(f"Loaded {len(g)} triples from {ttl_path}")
    return g


# ----------------------------
# 2) Build schema summary
# ----------------------------
def get_prefix_block(g: Graph) -> str:
    defaults = {
        "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd":  "http://www.w3.org/2001/XMLSchema#",
        "owl":  "http://www.w3.org/2002/07/owl#",
        "wd":   "http://www.wikidata.org/entity/",
        "wdt":  "http://www.wikidata.org/prop/direct/",
    }
    ns_map = {p: str(ns) for p, ns in g.namespace_manager.namespaces()}
    for k, v in defaults.items():
        ns_map.setdefault(k, v)
    return "\n".join(sorted(f"PREFIX {p}: <{ns}>" for p, ns in ns_map.items()))


def list_distinct_predicates(g: Graph, limit=MAX_PREDICATES) -> List[str]:
    return [str(row.p) for row in g.query(
        f"SELECT DISTINCT ?p WHERE {{ ?s ?p ?o . }} LIMIT {limit}")]


def list_distinct_classes(g: Graph, limit=MAX_CLASSES) -> List[str]:
    return [str(row.cls) for row in g.query(
        f"SELECT DISTINCT ?cls WHERE {{ ?s a ?cls . }} LIMIT {limit}")]


def sample_triples(g: Graph, limit=SAMPLE_TRIPLES) -> List[Tuple[str, str, str]]:
    return [(str(r.s), str(r.p), str(r.o)) for r in g.query(
        f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . }} LIMIT {limit}")]


def build_schema_summary(g: Graph) -> str:
    prefixes = get_prefix_block(g)
    preds = list_distinct_predicates(g)
    clss = list_distinct_classes(g)
    samples = sample_triples(g)
    pred_lines = "\n".join(f"- {p} ({PREDICATE_LABELS.get(p, '')})" for p in preds)
    cls_lines = "\n".join(f"- {c}" for c in clss)
    sample_lines = "\n".join(f"- {s} {p} {o}" for s, p, o in samples)
    return f"""
{prefixes}

# Available predicates (Wikidata direct properties):
{pred_lines}

# Classes / rdf:type:
{cls_lines}

# Sample triples from the graph:
{sample_lines}

# Domain: AI in Architecture
# Entities are Wikidata URIs: <http://www.wikidata.org/entity/Q...>
# Properties are Wikidata direct properties: <http://www.wikidata.org/prop/direct/P...>
""".strip()


# ----------------------------
# 3) NL → SPARQL
# ----------------------------
SPARQL_INSTRUCTIONS = """
You are a SPARQL generator for a knowledge base about AI in Architecture.
Convert the QUESTION into a valid SPARQL 1.1 SELECT query.

Strict rules:
- Return ONLY one valid SPARQL SELECT query in a ```sparql code block.
- SELECT clause must contain ONLY variables like ?type, ?part, ?country, ?entity.
- Never put wd:Q... or wdt:P... directly after SELECT.
- Always include PREFIX declarations.
- instance of / type → wdt:P31
- has part / parts → wdt:P527
- part of → wdt:P361
- country → wdt:P17
- before / follows → wdt:P155
- after / followed by → wdt:P156
- developer / developed by → wdt:P178
- employer → wdt:P108
- influenced by → wdt:P737
- creator → wdt:P170
- Always add LIMIT 20.
- NEVER use FILTER with custom functions like is_instance_of().

Example:
```sparql
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?type WHERE {
  wd:Q2283 wdt:P31 ?type .
}
LIMIT 20
```
"""


def detect_entity_qid(question: str) -> str:
    """Detect entity QID from question keywords."""
    q = question.lower()
    # Check for explicit QID
    match = re.search(r'\bq(\d+)\b', q)
    if match:
        return f"Q{match.group(1)}"
    # Check for known entity names
    for name, qid in ENTITY_MAP.items():
        if name in q:
            return qid
    return "Q35014"  # default fallback


def build_fallback_sparql(question: str) -> str:
    """Build a simple valid SPARQL query based on question keywords."""
    q = question.lower()
    entity = detect_entity_qid(question)

    if "instance of" in q or ("type" in q and "instance" in q):
        predicate, var = "P31", "?type"
    elif "has part" in q or ("parts" in q and "of" not in q):
        predicate, var = "P527", "?part"
    elif "part of" in q:
        predicate, var = "P361", "?whole"
    elif "country" in q:
        predicate, var = "P17", "?country"
    elif "before" in q or "follows" in q:
        predicate, var = "P155", "?before"
    elif "after" in q or "followed by" in q:
        predicate, var = "P156", "?after"
    elif "developer" in q or "developed by" in q:
        predicate, var = "P178", "?developer"
    elif "employer" in q or "works at" in q:
        predicate, var = "P108", "?employer"
    elif "influenced" in q:
        predicate, var = "P737", "?influencer"
    elif "creator" in q or "created by" in q:
        predicate, var = "P170", "?creator"
    elif "subclass" in q:
        predicate, var = "P279", "?subclass"
    elif "member" in q:
        predicate, var = "P463", "?member"
    else:
        # Generic: return all triples for the entity
        return f"""PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?p ?o WHERE {{
  wd:{entity} ?p ?o .
}}
LIMIT 20"""

    return f"""PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT {var} WHERE {{
  wd:{entity} wdt:{predicate} {var} .
}}
LIMIT 20"""


def validate_basic_sparql(query: str) -> str:
    query = query.strip()
    if "PREFIX wd:" not in query:
        query = "PREFIX wd: <http://www.wikidata.org/entity/>\n" + query
    if "PREFIX wdt:" not in query:
        query = "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n" + query
    # Fix common invalid SELECT patterns
    for bad, good in [
        ("SELECT wd:", "SELECT ?entity # was: wd:"),
        ("SELECT wdt:", "SELECT ?predicate # was: wdt:"),
        ("FILTER (is_instance_of", "# removed invalid filter"),
        ("FILTER(is_instance_of", "# removed invalid filter"),
    ]:
        if bad in query:
            return build_fallback_sparql("instance of")
    return query


CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sparql_from_text(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    if "SELECT" in text.upper():
        lines = text.strip().split("\n")
        sparql_lines = []
        in_query = False
        for line in lines:
            if "SELECT" in line.upper() or in_query:
                in_query = True
                sparql_lines.append(line)
        return "\n".join(sparql_lines).strip()
    return text.strip()


def generate_sparql(question: str, schema_summary: str) -> str:
    prompt = f"""{SPARQL_INSTRUCTIONS}
SCHEMA SUMMARY:
{schema_summary}

QUESTION:
{question}

Return only the SPARQL query in a code block.
"""
    raw = ask_local_llm(prompt)
    print("\n[Raw LLM Output]")
    print(raw)

    # If LLM error, use fallback directly
    if "[LLM Error" in raw:
        print("[Using fallback SPARQL — LLM unavailable]")
        return build_fallback_sparql(question)

    query = extract_sparql_from_text(raw)
    query = validate_basic_sparql(query)

    # If query looks empty or broken, use fallback
    if not query or len(query) < 20 or "SELECT" not in query.upper():
        print("[Using fallback SPARQL — LLM output invalid]")
        return build_fallback_sparql(question)

    return query


# ----------------------------
# 4) Execute SPARQL + self-repair
# ----------------------------
def run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple]]:
    res = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows = [tuple(str(cell) for cell in r) for r in res]
    return vars_, rows


REPAIR_INSTRUCTIONS = """
The previous SPARQL failed. Return a corrected SPARQL 1.1 SELECT query.
- Use only wd: and wdt: prefixes.
- No FILTER with custom functions.
- Keep it simple.
- Return only a ```sparql code block.
"""


def repair_sparql(schema_summary: str, question: str, bad_query: str, error_msg: str) -> str:
    prompt = f"""{REPAIR_INSTRUCTIONS}
QUESTION: {question}
BAD SPARQL: {bad_query}
ERROR: {error_msg}
Return only the corrected SPARQL in a code block.
"""
    raw = ask_local_llm(prompt)
    result = extract_sparql_from_text(raw)
    if not result or "[LLM Error" in result or "SELECT" not in result.upper():
        return build_fallback_sparql(question)
    return result


def answer_with_sparql_generation(g: Graph, schema_summary: str, question: str,
                                   try_repair: bool = True) -> dict:
    sparql = generate_sparql(question, schema_summary)
    try:
        vars_, rows = run_sparql(g, sparql)
        return {"query": sparql, "vars": vars_, "rows": rows, "repaired": False, "error": None}
    except Exception as e:
        err = str(e)
        print(f"\n[SPARQL failed — trying self-repair...] Error: {err}")
        if try_repair:
            repaired = repair_sparql(schema_summary, question, sparql, err)
            repaired = validate_basic_sparql(repaired)
            try:
                vars_, rows = run_sparql(g, repaired)
                return {"query": repaired, "vars": vars_, "rows": rows, "repaired": True, "error": None}
            except Exception as e2:
                # Last resort: force fallback
                fallback = build_fallback_sparql(question)
                try:
                    vars_, rows = run_sparql(g, fallback)
                    return {"query": fallback, "vars": vars_, "rows": rows, "repaired": True, "error": None}
                except Exception as e3:
                    return {"query": fallback, "vars": [], "rows": [], "repaired": True, "error": str(e3)}
        return {"query": sparql, "vars": [], "rows": [], "repaired": False, "error": err}


# ----------------------------
# 5) Baseline — no RAG
# ----------------------------
def answer_no_rag(question: str) -> str:
    return ask_local_llm(f"Answer this question briefly:\n\n{question}")


# ----------------------------
# 6) Pretty print
# ----------------------------
def pretty_print_result(result: dict):
    if result.get("error"):
        print("\n[Execution Error]", result["error"])
    print("\n[SPARQL Query Used]")
    print(result["query"])
    print("\n[Repaired?]", result["repaired"])
    vars_ = result.get("vars", [])
    rows = result.get("rows", [])
    if not rows:
        print("\n[No rows returned]")
        return
    print("\n[Results]")
    print(" | ".join(vars_))
    for r in rows[:20]:
        print(" | ".join(r))
    if len(rows) > 20:
        print(f"... (showing 20 of {len(rows)})")


# ----------------------------
# 7) CLI demo
# ----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Chatbot — AI in Architecture Knowledge Base")
    print(f"Graph: {TTL_FILE}")
    print(f"Model: {GEMMA_MODEL} via Ollama")
    print("=" * 60)

    g = load_graph(TTL_FILE)
    schema = build_schema_summary(g)


    while True:
        q = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() in ["quit", "exit", "q"]:
            break
        print("\n--- Baseline (No RAG) ---")
        print(answer_no_rag(q))
        print("\n--- SPARQL-generation RAG (gemma3:1b + rdflib) ---")
        result = answer_with_sparql_generation(g, schema, q, try_repair=True)
        pretty_print_result(result)
