import asyncio
import hashlib
import json
import math
import os
import random
import re
import time
import uuid
import sqlite3
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator

# Database Helpers
def get_db_connection():
    db_path = os.getenv("DATABASE_PATH", "production_llm.db")
    dir_name = os.path.dirname(db_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                name TEXT,
                version TEXT,
                template TEXT,
                model TEXT,
                max_output_tokens INTEGER,
                timestamp REAL,
                is_active INTEGER,
                rolled_back INTEGER,
                PRIMARY KEY (name, version)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                request_id TEXT PRIMARY KEY,
                user_id TEXT,
                timestamp TEXT,
                prompt_template TEXT,
                prompt_version TEXT,
                model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                latency_ms REAL,
                cache_hit INTEGER,
                guardrail_input_pass INTEGER,
                guardrail_output_pass INTEGER,
                cost_usd REAL,
                retrieval_latency_ms REAL,
                faithfulness REAL,
                error TEXT,
                rating REAL,
                tools_used TEXT
            );
        """)
        conn.commit()
    finally:
        conn.close()

# Structured JSON Logging Setup
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "filename": record.filename,
            "lineno": record.lineno
        }
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            ctx = span.get_span_context()
            log_data["trace_id"] = f"{ctx.trace_id:032x}"
            log_data["span_id"] = f"{ctx.span_id:016x}"
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def setup_logging():
    if os.getenv("ENABLE_JSON_LOGGING", "true").lower() == "true":
        root_logger = logging.getLogger()
        handler = logging.StreamHandler()
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
            
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        
        for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
            logger = logging.getLogger(logger_name)
            for h in logger.handlers[:]:
                logger.removeHandler(h)
            logger.addHandler(handler)
            logger.propagate = False

setup_logging()

# OpenTelemetry Tracing Setup
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter, SpanExporter, SpanExportResult, BatchSpanProcessor

class InMemorySpanExporter(SpanExporter):
    def __init__(self):
        self.spans = defaultdict(list)

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            trace_id = span.context.trace_id
            self.spans[trace_id].append(span)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

# Initialize global tracer provider and processors
provider = TracerProvider()
in_memory_exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))

if os.getenv("ENABLE_CONSOLE_TRACES", "true").lower() == "true":
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        endpoint_url = otlp_endpoint if otlp_endpoint.endswith("/v1/traces") else f"{otlp_endpoint}/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint_url)))
except ImportError:
    pass

trace.set_tracer_provider(provider)
tracer = trace.get_tracer("production_llm")

def print_trace_summary(trace_id):
    spans = in_memory_exporter.spans.get(trace_id, [])
    if not spans:
        return
    
    # Find root span (parent is None)
    root = None
    children = []
    for s in spans:
        if s.parent is None:
            root = s
        else:
            children.append(s)
            
    if not root:
        root = min(spans, key=lambda x: x.start_time)
        children = [s for s in spans if s != root]
        
    children.sort(key=lambda x: x.start_time)
    
    total_duration_ns = root.end_time - root.start_time
    total_duration_ms = total_duration_ns / 1_000_000.0
    
    print("\n" + "=" * 75)
    print(f"OPENTELEMETRY TRACE SUMMARY | Trace ID: {trace_id:032x}")
    print(f"Total Request Latency: {total_duration_ms:.2f}ms")
    print("-" * 75)
    print(f"  [Span] {root.name}: {total_duration_ms:.2f}ms (100.0%)")
    for child in children:
        child_duration_ns = child.end_time - child.start_time
        child_duration_ms = child_duration_ns / 1_000_000.0
        pct = (child_duration_ns / total_duration_ns * 100) if total_duration_ns > 0 else 0.0
        print(f"    ├── [Span] {child.name}: {child_duration_ms:.2f}ms ({pct:.1f}%)")
        if child.attributes:
            attrs = ", ".join(f"{k}={v}" for k, v in child.attributes.items())
            print(f"        Attributes: {{{attrs}}}")
    print("=" * 75 + "\n")


try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

from structures import ModelName, PromptTemplate
from constants import (
    MODEL_PRICING,
    FALLBACK_CHAIN,
    PROMPT_TEMPLATES,
    AB_EXPERIMENTS,
    INJECTION_PATTERNS,
    PII_PATTERNS,
    BANNED_OUTPUT_PATTERNS,
    SIMULATED_RESPONSES,
)
from tools import tool_registry, detect_tool_mock

from rag.chunker import Chunker
from rag.embeddings.tfidf import TFIDFEmbeder
from rag_docs import RAG_DOCUMENTS
import chromadb



@dataclass
class RequestLog:
    request_id: str
    user_id: str
    timestamp: str
    prompt_template: str
    prompt_version: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_hit: bool
    guardrail_input_pass: bool
    guardrail_output_pass: bool
    cost_usd: float
    retrieval_latency_ms: float = 0.0
    faithfulness: float | None = None
    error: str | None = None
    tools_used: list = field(default_factory=list)
    rating: float | None = None


@dataclass
class CostTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_requests: int = 0
    total_cache_hits: int = 0
    cost_by_user: dict = field(default_factory=lambda: defaultdict(float))
    cost_by_model: dict = field(default_factory=lambda: defaultdict(float))
    daily_user_costs: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    daily_total_cost: dict = field(default_factory=lambda: defaultdict(float))

    def record(self, user_id, model, input_tokens, output_tokens, cost, timestamp_str=None):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.total_requests += 1
        self.cost_by_user[user_id] += cost
        self.cost_by_model[model] += cost
        
        if timestamp_str:
            try:
                date_str = datetime.fromisoformat(timestamp_str).date().isoformat()
            except Exception:
                date_str = datetime.now(timezone.utc).date().isoformat()
        else:
            date_str = datetime.now(timezone.utc).date().isoformat()
            
        self.daily_user_costs[date_str][user_id] += cost
        self.daily_total_cost[date_str] += cost

    def get_user_daily_cost(self, user_id, date_str=None) -> float:
        if date_str is None:
            date_str = datetime.now(timezone.utc).date().isoformat()
        return self.daily_user_costs[date_str][user_id]

    def get_total_daily_cost(self, date_str=None) -> float:
        if date_str is None:
            date_str = datetime.now(timezone.utc).date().isoformat()
        return self.daily_total_cost[date_str]

    def summary(self):
        avg_cost = self.total_cost_usd / max(self.total_requests, 1)
        cache_rate = self.total_cache_hits / max(self.total_requests, 1) * 100
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_cost_per_request": round(avg_cost, 6),
            "cache_hit_rate_pct": round(cache_rate, 2),
            "cost_by_model": dict(self.cost_by_model),
            "top_users_by_cost": dict(
                sorted(self.cost_by_user.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }








class PromptVersion:
    def __init__(self, name: str, version: str, template: str, model: ModelName, max_output_tokens: int = 1024, timestamp: float = None):
        self.name = name
        self.version = version
        self.template = template
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timestamp = timestamp or time.time()
        self.is_active = True
        self.rolled_back = False

class PromptRegistry:
    def __init__(self):
        self._prompts = {}
        self.rollback_min_requests = 100

    def register(self, name: str, version: str, template: str, model: ModelName, max_output_tokens: int = 1024, timestamp: float = None, save_to_db: bool = True):
        if name not in self._prompts:
            self._prompts[name] = {}
        pv = PromptVersion(name, version, template, model, max_output_tokens, timestamp)
        self._prompts[name][version] = pv
        
        if save_to_db:
            conn = get_db_connection()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO prompts (name, version, template, model, max_output_tokens, timestamp, is_active, rolled_back)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (pv.name, pv.version, pv.template, pv.model.value, pv.max_output_tokens, pv.timestamp, int(pv.is_active), int(pv.rolled_back)))
                conn.commit()
            except Exception as e:
                import logging
                logging.error(f"Error saving registered prompt {name}:{version} to DB: {e}")
            finally:
                conn.close()
        return pv

    def get_prompt(self, name: str, version: str) -> PromptVersion:
        return self._prompts.get(name, {}).get(version)

    def get_versions_dict(self, name: str) -> dict:
        return self._prompts.get(name)

    def get_previous_version(self, template_name: str, current_version: str) -> str:
        versions = self._prompts.get(template_name, {})
        sorted_vers = sorted(versions.values(), key=lambda x: x.timestamp)
        idx = -1
        for i, pv in enumerate(sorted_vers):
            if pv.version == current_version:
                idx = i
                break
        if idx <= 0:
            return None
        for i in range(idx - 1, -1, -1):
            pv = sorted_vers[i]
            if pv.is_active and not pv.rolled_back:
                return pv.version
        return None

    def check_and_trigger_rollback(self, template_name: str, version: str, request_logs: list) -> bool:
        pv = self.get_prompt(template_name, version)
        if not pv or pv.rolled_back or not pv.is_active:
            return False

        prev_version = self.get_previous_version(template_name, version)
        if not prev_version:
            return False

        curr_logs = [l for l in request_logs if l.prompt_template == template_name and l.prompt_version == version]
        if len(curr_logs) < self.rollback_min_requests:
            return False

        last_n_curr = curr_logs[-self.rollback_min_requests:]
        curr_errors = sum(1 for l in last_n_curr if l.error is not None)
        curr_error_rate = curr_errors / len(last_n_curr)

        prev_logs = [l for l in request_logs if l.prompt_template == template_name and l.prompt_version == prev_version]
        if not prev_logs:
            prev_error_rate = 0.0
        else:
            last_n_prev = prev_logs[-self.rollback_min_requests:]
            prev_errors = sum(1 for l in last_n_prev if l.error is not None)
            prev_error_rate = prev_errors / len(last_n_prev)

        if curr_error_rate >= 2.0 * prev_error_rate and curr_error_rate > 0.0:
            pv.rolled_back = True
            pv.is_active = False
            
            conn = get_db_connection()
            try:
                conn.execute("""
                    UPDATE prompts SET rolled_back = ?, is_active = ? WHERE name = ? AND version = ?
                """, (int(pv.rolled_back), int(pv.is_active), pv.name, pv.version))
                conn.commit()
            except Exception as e:
                import logging
                logging.error(f"Error updating rolled back prompt status in DB: {e}")
            finally:
                conn.close()
                
            import logging
            logging.warning(
                f"AUTOMATIC ROLLBACK TRIGGERED: Prompt version {template_name}:{version} "
                f"has error rate {curr_error_rate:.4f} which is >= 2x the previous version "
                f"{prev_version} error rate ({prev_error_rate:.4f}) over the last {len(last_n_curr)} requests."
            )
            return True
        return False

def load_prompts_from_db(registry: PromptRegistry):
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name, version, template, model, max_output_tokens, timestamp, is_active, rolled_back FROM prompts")
        rows = cursor.fetchall()
        
        if not rows:
            for name, versions in PROMPT_TEMPLATES.items():
                for ver, tmpl in versions.items():
                    registry.register(
                        name=tmpl.name,
                        version=tmpl.version,
                        template=tmpl.template,
                        model=tmpl.model,
                        max_output_tokens=tmpl.max_output_tokens,
                        save_to_db=True
                    )
            return
            
        for name, version, template, model_str, max_output_tokens, timestamp, is_active, rolled_back in rows:
            from structures import ModelName
            try:
                model_enum = ModelName(model_str)
            except ValueError:
                model_enum = ModelName.GPT_4O_MINI
            
            pv = registry.register(
                name=name,
                version=version,
                template=template,
                model=model_enum,
                max_output_tokens=max_output_tokens,
                timestamp=timestamp,
                save_to_db=False
            )
            pv.is_active = bool(is_active)
            pv.rolled_back = bool(rolled_back)
    except Exception as e:
        import logging
        logging.error(f"Error loading prompts from DB: {e}")
    finally:
        conn.close()

prompt_registry = PromptRegistry()
load_prompts_from_db(prompt_registry)

def select_prompt(template_name, user_id, variables):
    versions = prompt_registry.get_versions_dict(template_name)
    if not versions:
        raise ValueError(f"Unknown template: {template_name}")

    version = "v1"
    for exp_name, exp in AB_EXPERIMENTS.items():
        if exp["template"] == template_name:
            bucket = int(hashlib.md5(f"{user_id}:{exp_name}".encode()).hexdigest(), 16) % 100
            if bucket < exp["traffic_pct"]:
                version = exp["variant"]
            else:
                version = exp["control"]
            break

    pv = versions.get(version)
    if not pv or pv.rolled_back or not pv.is_active:
        if version != "v1" and versions.get("v1") and not versions["v1"].rolled_back:
            version = "v1"
        else:
            active_vers = [v for v, p in versions.items() if p.is_active and not p.rolled_back]
            version = active_vers[0] if active_vers else "v1"

    template = versions.get(version)
    rendered = template.template.format(**variables)
    return template, rendered


def simple_embedding(text, dim=64):
    h = hashlib.sha256(text.lower().strip().encode()).hexdigest()
    raw = [int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), dim * 2), 2)]
    while len(raw) < dim:
        ext = hashlib.sha256(f"{text}_{len(raw)}".encode()).hexdigest()
        raw.extend([int(ext[i:i+2], 16) / 255.0 for i in range(0, min(len(ext), (dim - len(raw)) * 2), 2)])
    raw = raw[:dim]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm if norm > 0 else 0.0 for x in raw]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    def __init__(self, similarity_threshold=0.92, max_entries=10000, ttl_seconds=3600):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self.entries = []
        self.hits = 0
        self.misses = 0

    def get(self, query):
        query_emb = simple_embedding(query)
        now = time.time()

        best_score = 0.0
        best_entry = None

        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            score = cosine_similarity(query_emb, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            self.hits += 1
            return {
                "response": best_entry["response"],
                "tools_used": best_entry.get("tools_used", []),
                "similarity": round(best_score, 4),
                "original_query": best_entry["query"],
                "cached_at": best_entry["timestamp"],
            }

        self.misses += 1
        return None

    def put(self, query, response, tools_used=None):
        if len(self.entries) >= self.max_entries:
            self.entries.sort(key=lambda e: e["timestamp"])
            self.entries = self.entries[len(self.entries) // 4:]

        self.entries.append({
            "query": query,
            "embedding": simple_embedding(query),
            "response": response,
            "tools_used": tools_used or [],
            "timestamp": time.time(),
        })

    def stats(self):
        total = self.hits + self.misses
        return {
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(self.hits / max(total, 1) * 100, 2),
        }





@dataclass
class GuardrailResult:
    passed: bool
    blocked_reason: str | None = None
    pii_detected: list = field(default_factory=list)
    modified_text: str | None = None


def check_input_guardrails(text):
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailResult(
                passed=False,
                blocked_reason="Potential prompt injection detected",
            )

    pii_found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            pii_found.append(pii_type)

    if pii_found:
        redacted = text
        for pii_type, pattern in PII_PATTERNS.items():
            redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
        return GuardrailResult(
            passed=True,
            pii_detected=pii_found,
            modified_text=redacted,
        )

    return GuardrailResult(passed=True)


def check_output_guardrails(text):
    for pattern in BANNED_OUTPUT_PATTERNS:
        if re.search(pattern, text):
            return GuardrailResult(
                passed=False,
                blocked_reason="Response contained potentially unsafe content",
            )
    return GuardrailResult(passed=True)


def estimate_tokens(text):
    return max(1, len(text.split()) * 4 // 3)


def calculate_cost(model, input_tokens, output_tokens):
    pricing = MODEL_PRICING.get(model, MODEL_PRICING[ModelName.GPT_4O])
    input_cost = input_tokens / 1_000_000 * pricing["input"]
    output_cost = output_tokens / 1_000_000 * pricing["output"]
    return round(input_cost + output_cost, 8)



async def call_openai(prompt, model="gpt-4o"):
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        yield delta


async def call_anthropic(prompt, model="claude-sonnet-4-20250514"):
    client = anthropic.AsyncAnthropic()
    async with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def call_llm_with_retry(prompt, model, max_retries=3):
    is_mockup = os.getenv("MOCKUP_MODE", "").lower() in ("true", "1")
    
    if not is_mockup:
        if model in (ModelName.GPT_4O, ModelName.GPT_4O_MINI):
            if openai is None or not os.getenv("OPENAI_API_KEY", "").strip():
                is_mockup = True
        elif model == ModelName.CLAUDE_SONNET:
            if anthropic is None or not os.getenv("ANTHROPIC_API_KEY", "").strip():
                is_mockup = True
        else:
            is_mockup = True

    exceptions_to_catch = [ConnectionError, TimeoutError]
    if openai is not None:
        exceptions_to_catch.append(openai.APIError)
    if anthropic is not None:
        exceptions_to_catch.append(anthropic.APIError)
    exceptions_to_catch = tuple(exceptions_to_catch)

    for attempt in range(max_retries + 1):
        try:
            with tracer.start_as_current_span("llm_call") as span:
                span.set_attribute("llm.model", model.value if hasattr(model, "value") else str(model))
                span.set_attribute("llm.attempt", attempt)
                if is_mockup:
                    failure_chance = 0.15 if attempt == 0 else 0.05
                    if random.random() < failure_chance:
                        raise ConnectionError(f"API error from {model.value}: 500 Internal Server Error")

                    await asyncio.sleep(random.uniform(0.1, 0.3))

                    if "code" in prompt.lower() or "review" in prompt.lower():
                        response_text = SIMULATED_RESPONSES["code_review"]
                    elif "no context provided" in prompt.lower():
                        response_text = SIMULATED_RESPONSES["rag_no_info"]
                    elif "ceo" in prompt.lower() and "discount" in prompt.lower():
                        response_text = SIMULATED_RESPONSES["rag_ceo"]
                    elif "context" in prompt.lower():
                        response_text = SIMULATED_RESPONSES["rag"]
                    else:
                        response_text = SIMULATED_RESPONSES["general"]
                else:
                    response_text = ""
                    if model in (ModelName.GPT_4O, ModelName.GPT_4O_MINI):
                        async for delta in call_openai(prompt, model.value):
                            response_text += delta
                    elif model == ModelName.CLAUDE_SONNET:
                        async for delta in call_anthropic(prompt, model.value):
                            response_text += delta
                    else:
                        raise ValueError(f"Unknown model for real execution: {model}")

                span.set_attribute("llm.input_tokens", estimate_tokens(prompt))
                span.set_attribute("llm.output_tokens", estimate_tokens(response_text))
                return {
                    "text": response_text,
                    "model": model.value,
                    "input_tokens": estimate_tokens(prompt),
                    "output_tokens": estimate_tokens(response_text),
                }

        except exceptions_to_catch as e:
            if attempt < max_retries:
                backoff = min(2 ** attempt + random.uniform(0, 1), 10)
                await asyncio.sleep(backoff)
            else:
                raise

    raise ConnectionError(f"All {max_retries} retries exhausted for {model.value}")


async def call_with_fallback(prompt, preferred_model=None, allowed_models=None):
    chain = list(allowed_models) if allowed_models is not None else list(FALLBACK_CHAIN)
    if preferred_model and preferred_model in chain:
        chain.remove(preferred_model)
        chain.insert(0, preferred_model)

    last_error = None
    for model in chain:
        try:
            return await call_llm_with_retry(prompt, model)
        except ConnectionError as e:
            last_error = e
            continue

    return {
        "text": "I apologize, but I am temporarily unable to process your request. Please try again in a moment.",
        "model": "fallback",
        "input_tokens": estimate_tokens(prompt),
        "output_tokens": 20,
        "error": str(last_error),
    }


async def stream_response(text):
    words = text.split()
    for i, word in enumerate(words):
        token = word if i == 0 else " " + word
        yield token
        await asyncio.sleep(random.uniform(0.02, 0.08))


TOOL_DETECTION_PROMPT = """You are an API router. Determine if the user's query requires external data by calling one of the following tools:
- "weather": requires location (e.g., city/state).
- "calculate": requires a mathematical expression to evaluate (e.g., 2+2, 5*3).
- "search": requires a query string for general web/external search.

Respond strictly with a JSON object. Do not include any explanations or markdown formatting outside the JSON block.
If no tool is needed, respond with:
{{"tool": null, "arguments": {{}}}}

If a tool is needed, respond with:
{{"tool": "<tool_name>", "arguments": {{"<arg_name>": "<arg_value>"}}}}

User query: {query}
Response:"""

async def detect_tool_llm(query: str) -> dict:
    prompt = TOOL_DETECTION_PROMPT.format(query=query)
    try:
        result = await call_with_fallback(prompt, preferred_model=ModelName.GPT_4O_MINI)
        text = result["text"].strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                text = "\n".join(lines[1:-1])
        data = json.loads(text.strip())
        if isinstance(data, dict) and "tool" in data:
            return data
    except Exception:
        pass
    return {"tool": None, "arguments": {}}


class ProductionLLMService:
    def __init__(self):
        self.cache = SemanticCache(similarity_threshold=0.92, ttl_seconds=3600)
        self.cost_tracker = CostTracker()
        self.request_logs = []
        self.eval_results = []
        self.tool_registry = tool_registry
        self.prompt_registry = prompt_registry

        # RAG integration
        self.rag_documents = RAG_DOCUMENTS
        self.chunker = Chunker(max_tokens=64, overlap=10)
        self.rag_chunks = []
        self.rag_embeddings = []

        # Populate vector store
        for idx, doc in enumerate(self.rag_documents):
            source = f"doc_{idx+1}.md"
            doc_chunks = self.chunker.chunk_text(doc)
            for chunk_words in doc_chunks:
                chunk_text = " ".join(chunk_words)
                self.rag_chunks.append({
                    "text": chunk_text,
                    "source": source,
                })

        # Build TF-IDF Embedder
        self.embeder = TFIDFEmbeder()
        chunk_texts = [c["text"] for c in self.rag_chunks]
        self.embeder.build_vocabulary(chunk_texts)
        self.embeder.compute_idf(chunk_texts)
        self.rag_embeddings = [self.embeder.embed(text) for text in chunk_texts]

        # Initialize ChromaDB persistent client
        chroma_db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            "acme_docs",
            metadata={"hnsw:space": "cosine"}
        )
        if self.collection.count() == 0:
            self.collection.add(
                ids=[f"chunk_{i}" for i in range(len(self.rag_chunks))],
                embeddings=self.rag_embeddings,
                documents=chunk_texts,
                metadatas=[{"source": c["source"]} for c in self.rag_chunks]
            )
            
        # Reconstruct state from SQLite DB
        self.request_logs = []
        self._load_logs_and_reconstruct_state()

    def _load_logs_and_reconstruct_state(self):
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT request_id, user_id, timestamp, prompt_template, prompt_version, model,
                       input_tokens, output_tokens, latency_ms, cache_hit, guardrail_input_pass,
                       guardrail_output_pass, cost_usd, retrieval_latency_ms, faithfulness,
                       error, rating, tools_used FROM request_logs
            """)
            rows = cursor.fetchall()
            for row in rows:
                tools_list = []
                if row[17]:
                    try:
                        tools_list = json.loads(row[17])
                    except Exception:
                        pass
                
                log = RequestLog(
                    request_id=row[0],
                    user_id=row[1],
                    timestamp=row[2],
                    prompt_template=row[3],
                    prompt_version=row[4],
                    model=row[5],
                    input_tokens=row[6],
                    output_tokens=row[7],
                    latency_ms=row[8],
                    cache_hit=bool(row[9]),
                    guardrail_input_pass=bool(row[10]),
                    guardrail_output_pass=bool(row[11]),
                    cost_usd=row[12],
                    retrieval_latency_ms=row[13],
                    faithfulness=row[14],
                    error=row[15],
                    rating=row[16],
                    tools_used=tools_list
                )
                self.request_logs.append(log)
                
                if log.cache_hit:
                    self.cost_tracker.total_cache_hits += 1
                
                self.cost_tracker.record(
                    user_id=log.user_id,
                    model=log.model,
                    input_tokens=log.input_tokens,
                    output_tokens=log.output_tokens,
                    cost=log.cost_usd,
                    timestamp_str=log.timestamp
                )
        except Exception as e:
            import logging
            logging.error(f"Error loading request logs from DB: {e}")
        finally:
            conn.close()

    def save_log_to_db(self, log: RequestLog):
        conn = get_db_connection()
        try:
            tools_json = json.dumps(log.tools_used)
            conn.execute("""
                INSERT INTO request_logs (
                    request_id, user_id, timestamp, prompt_template, prompt_version, model,
                    input_tokens, output_tokens, latency_ms, cache_hit, guardrail_input_pass,
                    guardrail_output_pass, cost_usd, retrieval_latency_ms, faithfulness,
                    error, rating, tools_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log.request_id, log.user_id, log.timestamp, log.prompt_template, log.prompt_version,
                log.model, log.input_tokens, log.output_tokens, log.latency_ms, int(log.cache_hit),
                int(log.guardrail_input_pass), int(log.guardrail_output_pass), log.cost_usd,
                log.retrieval_latency_ms, log.faithfulness, log.error, log.rating, tools_json
            ))
            conn.commit()
        except Exception as e:
            import logging
            logging.error(f"Error saving request log {log.request_id} to DB: {e}")
        finally:
            conn.close()

    async def handle_request(self, user_id, query, template_name="general_chat", variables=None):
        trace_id = None
        try:
            with tracer.start_as_current_span("handle_request") as root_span:
                trace_id = root_span.get_span_context().trace_id
                request_id = str(uuid.uuid4())[:12]
                start_time = time.time()
                variables = variables or {}
                variables["query"] = query

                with tracer.start_as_current_span("input_guardrails") as span:
                    input_check = check_input_guardrails(query)
                    span.set_attribute("guardrails.input.passed", input_check.passed)

                if not input_check.passed:
                    return self._blocked_response(request_id, user_id, template_name, input_check, start_time)

                # Get daily costs
                date_str = datetime.now(timezone.utc).date().isoformat()
                user_daily_cost = self.cost_tracker.get_user_daily_cost(user_id, date_str)
                total_daily_cost = self.cost_tracker.get_total_daily_cost(date_str)
                is_emergency = total_daily_cost > 100.0

                original_effective_query = input_check.modified_text or query
                effective_query = original_effective_query
                if input_check.modified_text:
                    variables["query"] = effective_query

                with tracer.start_as_current_span("cache_lookup") as span:
                    cached = self.cache.get(effective_query)
                    span.set_attribute("cache.hit", cached is not None)

                if cached:
                    self.cost_tracker.total_cache_hits += 1
                    log = RequestLog(
                        request_id=request_id,
                        user_id=user_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        prompt_template=template_name,
                        prompt_version="cached",
                        model="cache",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=round((time.time() - start_time) * 1000, 2),
                        cache_hit=True,
                        guardrail_input_pass=True,
                        guardrail_output_pass=True,
                        cost_usd=0.0,
                        retrieval_latency_ms=0.0,
                        faithfulness=None,
                        tools_used=cached.get("tools_used", [])
                    )
                    self.request_logs.append(log)
                    self.save_log_to_db(log)
                    self.cost_tracker.record(user_id, "cache", 0, 0, 0.0)
                    return {
                        "request_id": request_id,
                        "response": cached["response"],
                        "cache_hit": True,
                        "similarity": cached["similarity"],
                        "latency_ms": log.latency_ms,
                        "retrieval_latency_ms": 0.0,
                        "faithfulness": None,
                        "cost_usd": 0.0,
                        "tools_used": cached.get("tools_used", []),
                    }

                # Tool detection and execution pipeline
                tools_used = []
                is_mockup = os.getenv("MOCKUP_MODE", "").lower() in ("true", "1") or not (
                    (os.getenv("OPENAI_API_KEY") and openai is not None) or 
                    (os.getenv("ANTHROPIC_API_KEY") and anthropic is not None)
                )

                tool_call = None
                if is_mockup:
                    tool_call = detect_tool_mock(effective_query)
                else:
                    tool_call = await detect_tool_llm(effective_query)

                if tool_call and tool_call.get("tool"):
                    tool_name = tool_call["tool"]
                    tool_args = tool_call.get("arguments", {})
                    try:
                        tool_result = self.tool_registry.execute(tool_name, **tool_args)
                        tools_used.append({
                            "name": tool_name,
                            "arguments": tool_args,
                            "result": tool_result
                        })
                        # Inject tool result into the prompt query
                        effective_query = (
                            f"{effective_query}\n\n"
                            f"[Tool Result]\n"
                            f"Tool: {tool_name}\n"
                            f"Arguments: {json.dumps(tool_args)}\n"
                            f"Result: {tool_result}"
                        )
                        variables["query"] = effective_query
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_name}: {str(e)}"
                        tools_used.append({
                            "name": tool_name,
                            "arguments": tool_args,
                            "result": error_msg
                        })

                # RAG retrieval logic on cache miss
                retrieval_latency_ms = 0.0
                retrieved_chunks = []
                is_rag = (template_name == "rag_answer")
                bypass_rag = variables.get("bypass_rag", False) if variables else False

                if is_rag:
                    retrieval_start = time.time()
                    query_emb = self.embeder.embed(original_effective_query)
                    results = self.collection.query(
                        query_embeddings=[query_emb],
                        n_results=3
                    )
                    retrieved_chunks = []
                    if results and results.get("documents") and results["documents"][0]:
                        for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
                            retrieved_chunks.append({
                                "text": doc_text,
                                "source": meta["source"]
                            })
                    retrieval_latency_ms = round((time.time() - retrieval_start) * 1000, 2)

                    if not bypass_rag:
                        context_str = "\n\n".join([f"Source: {c['source']}\nContent: {c['text']}" for c in retrieved_chunks])
                        variables["context"] = context_str
                    else:
                        if "context" not in variables:
                            variables["context"] = "No context provided."

                template, rendered_prompt = select_prompt(template_name, user_id, variables)
                input_tokens = estimate_tokens(rendered_prompt)

                # Check emergency mode size constraint
                if is_emergency and input_tokens > 2000:
                    import logging
                    logging.warning(f"Emergency mode rejection: Request has {input_tokens} input tokens, which exceeds the 2,000 limit.")
                    return self._emergency_blocked_response(
                        request_id, user_id, template_name,
                        "Emergency Mode: Request size exceeds 2,000 input tokens limit",
                        start_time
                    )

                # Cost routing rules
                if is_emergency:
                    target_model = ModelName.GPT_4O_MINI
                    allowed_models = [ModelName.GPT_4O_MINI]
                    import logging
                    logging.warning(f"Emergency mode active (Daily Cost: ${total_daily_cost:.4f})! Forcing gpt-4o-mini.")
                elif user_daily_cost > 0.50:
                    target_model = ModelName.GPT_4O_MINI
                    allowed_models = [ModelName.GPT_4O_MINI]
                    import logging
                    logging.warning(f"User budget warning: User {user_id} exceeded $0.50 daily limit (${user_daily_cost:.4f}). Downgrading to gpt-4o-mini.")
                else:
                    target_model = template.model
                    allowed_models = None

                result = await call_with_fallback(rendered_prompt, target_model, allowed_models=allowed_models)

                with tracer.start_as_current_span("output_guardrails") as span:
                    output_check = check_output_guardrails(result["text"])
                    span.set_attribute("guardrails.output.passed", output_check.passed)

                if not output_check.passed:
                    result["text"] = "I cannot provide that response as it was flagged by our safety system."
                    result["output_tokens"] = estimate_tokens(result["text"])

                with tracer.start_as_current_span("cost_calculation") as span:
                    cost = calculate_cost(
                        ModelName(result["model"]) if result["model"] != "fallback" else ModelName.GPT_4O_MINI,
                        result["input_tokens"],
                        result["output_tokens"],
                    )
                    span.set_attribute("cost.usd", cost)

                latency_ms = round((time.time() - start_time) * 1000, 2)

                # Faithfulness evaluation
                faithfulness = None
                if is_rag and not cached and not result.get("error"):
                    try:
                        from rag.evaluation import evaluate_faithfulness
                        retrieved_texts = [c["text"] for c in retrieved_chunks]
                        faithfulness, _ = evaluate_faithfulness(result["text"], retrieved_texts)
                    except Exception:
                        faithfulness = None

                log = RequestLog(
                    request_id=request_id,
                    user_id=user_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    prompt_template=template_name,
                    prompt_version=template.version,
                    model=result["model"],
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    latency_ms=latency_ms,
                    cache_hit=False,
                    guardrail_input_pass=True,
                    guardrail_output_pass=output_check.passed,
                    cost_usd=cost,
                    retrieval_latency_ms=retrieval_latency_ms,
                    faithfulness=faithfulness,
                    error=result.get("error"),
                    tools_used=tools_used
                )
                self.request_logs.append(log)
                self.save_log_to_db(log)
                self.cost_tracker.record(user_id, result["model"], result["input_tokens"], result["output_tokens"], cost)

                self.cache.put(original_effective_query, result["text"], tools_used=tools_used)

                self._log_eval(request_id, template_name, template.version, result, latency_ms)

                # Trigger automatic rollback check
                self.prompt_registry.check_and_trigger_rollback(template_name, template.version, self.request_logs)

                return {
                    "request_id": request_id,
                    "response": result["text"],
                    "model": result["model"],
                    "cache_hit": False,
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "latency_ms": latency_ms,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "faithfulness": faithfulness,
                    "cost_usd": cost,
                    "pii_detected": input_check.pii_detected,
                    "guardrail_output_pass": output_check.passed,
                    "tools_used": tools_used,
                }
        finally:
            if trace_id is not None:
                print_trace_summary(trace_id)

    async def handle_streaming_request(self, user_id, query, template_name="general_chat"):
        result = await self.handle_request(user_id, query, template_name)
        if result.get("cache_hit") or result.get("blocked"):
            return result

        tokens = []
        async for token in stream_response(result["response"]):
            tokens.append(token)
        result["streamed"] = True
        result["stream_tokens"] = len(tokens)
        return result

    def _emergency_blocked_response(self, request_id, user_id, template_name, reason, start_time):
        log = RequestLog(
            request_id=request_id,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_template=template_name,
            prompt_version="blocked_emergency",
            model="none",
            input_tokens=0,
            output_tokens=0,
            latency_ms=round((time.time() - start_time) * 1000, 2),
            cache_hit=False,
            guardrail_input_pass=True,
            guardrail_output_pass=True,
            cost_usd=0.0,
            error=reason,
            tools_used=[]
        )
        self.request_logs.append(log)
        self.save_log_to_db(log)
        return {
            "request_id": request_id,
            "blocked": True,
            "reason": reason,
            "latency_ms": log.latency_ms,
            "cost_usd": 0.0,
            "tools_used": [],
        }

    def _blocked_response(self, request_id, user_id, template_name, guardrail_result, start_time):
        log = RequestLog(
            request_id=request_id,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_template=template_name,
            prompt_version="blocked",
            model="none",
            input_tokens=0,
            output_tokens=0,
            latency_ms=round((time.time() - start_time) * 1000, 2),
            cache_hit=False,
            guardrail_input_pass=False,
            guardrail_output_pass=True,
            cost_usd=0.0,
            error=guardrail_result.blocked_reason,
            tools_used=[]
        )
        self.request_logs.append(log)
        self.save_log_to_db(log)
        return {
            "request_id": request_id,
            "blocked": True,
            "reason": guardrail_result.blocked_reason,
            "latency_ms": log.latency_ms,
            "cost_usd": 0.0,
            "tools_used": [],
        }

    def _log_eval(self, request_id, template_name, version, result, latency_ms):
        self.eval_results.append({
            "request_id": request_id,
            "template": template_name,
            "version": version,
            "model": result["model"],
            "output_length": len(result["text"]),
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_feedback(self, request_id: str, rating: float) -> bool:
        if not isinstance(rating, (int, float)) or not (1.0 <= rating <= 5.0):
            return False
        for log in self.request_logs:
            if log.request_id == request_id:
                log.rating = rating
                
                conn = get_db_connection()
                try:
                    conn.execute("UPDATE request_logs SET rating = ? WHERE request_id = ?", (rating, request_id))
                    conn.commit()
                except Exception as e:
                    import logging
                    logging.error(f"Error updating feedback rating for {request_id} in DB: {e}")
                finally:
                    conn.close()
                return True
        return False

    def get_prompt_metrics(self) -> dict:
        metrics = {}
        for template_name, versions in self.prompt_registry._prompts.items():
            for version, pv in versions.items():
                key = f"{template_name}:{version}"
                metrics[key] = {
                    "template_name": template_name,
                    "version": version,
                    "timestamp": pv.timestamp,
                    "is_active": pv.is_active,
                    "rolled_back": pv.rolled_back,
                    "total_requests": 0,
                    "total_errors": 0,
                    "error_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "avg_rating": None,
                    "ratings_count": 0
                }

        for log in self.request_logs:
            key = f"{log.prompt_template}:{log.prompt_version}"
            if key not in metrics:
                continue
            
            m = metrics[key]
            m["total_requests"] += 1
            if log.error is not None:
                m["total_errors"] += 1
            
            m["avg_latency_ms"] += log.latency_ms
            
            if log.rating is not None:
                if m["avg_rating"] is None:
                    m["avg_rating"] = 0.0
                m["avg_rating"] += log.rating
                m["ratings_count"] += 1

        for key, m in metrics.items():
            reqs = m["total_requests"]
            if reqs > 0:
                m["error_rate"] = round(m["total_errors"] / reqs, 4)
                m["avg_latency_ms"] = round(m["avg_latency_ms"] / reqs, 2)
            else:
                m["error_rate"] = 0.0
                m["avg_latency_ms"] = 0.0
            
            rc = m["ratings_count"]
            if rc > 0:
                m["avg_rating"] = round(m["avg_rating"] / rc, 2)
            else:
                m["avg_rating"] = None

        return metrics

    def health_check(self):
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache": self.cache.stats(),
            "cost": self.cost_tracker.summary(),
            "total_requests": len(self.request_logs),
            "eval_entries": len(self.eval_results),
        }
