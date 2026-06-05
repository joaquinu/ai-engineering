# Production LLM

# Objective
Wire together all the pieces (prompts, RAG, function calling, caching, guardrails) into a single production-ready service
Implement streaming token delivery, graceful error handling, and request timeout management
Build observability into the application: request logging, cost tracking, latency percentiles, and error rate dashboards
Deploy the application with health checks, rate limiting, and a fallback strategy for provider outages

# Problems to be solve
1. A user sends a 50,000-token document. Your context window overflows.
2. Two users ask the same question 4 seconds apart. You pay for both.
3. The API returns a 500 error at 2am. Your service crashes.
4. A user asks the model to generate SQL. The model outputs DROP TABLE users.
5. Your monthly bill hits $12,000 and you have no idea which feature caused it.
6. Response time averages 8 seconds. Users leave after 3.

# Architecture

[Client (web,mobile, API)] -> [API Gateway (Auth + rate limit)] -> [Guardrails (Input + Output)]

Input Check -> [Prompt Router Template Selection] -> [Semantic Cache - Embedding Lookup] ->(hit) [Response SSE Stream]
        (miss) -> []                                                                     ->(miss) [LLM Call Streaming]

Output Check -> [Eval Logger Quality Tracking] -> [Cost Tracker Token Accounting] ->(hit) [Response SSE Stream]
        (miss) -> []                                                              ->(miss) [LLM Call Streaming]


# The Stack


| Component | Technology | Purpose |
|-----------|------------|---------|
| API Server | FastAPI + Uvicorn | HTTP endpoints, SSE streaming, health checks |
| Prompt Templates | Jinja2 / string templates | Versioned prompt management with variable injection |
| Embeddings | text-embedding-3-small | Semantic similarity for cache and RAG |
| Vector Store | In-memory (prod: Pinecone/Qdrant) | Nearest neighbor search for context retrieval |
| Function Calling | Tool registry + JSON Schema | External data access, structured actions |
| Evaluation | Custom metrics + logging | Response quality, latency, accuracy tracking |
| Caching | Semantic cache (embedding-based) | Avoid redundant LLM calls, reduce cost and latency |
| Guardrails | Regex + classifier rules | Block prompt injection, PII, unsafe content |
| Cost Tracker | Token counter + pricing table | Per-request and aggregate cost accounting |
| Streaming | Server-Sent Events (SSE) | Token-by-token delivery, sub-second first token |


# Streaming

Three protocols for streaming:

| Protocol | Latency | Complexity | When to Use |
|----------|---------|------------|-------------|
| Server-Sent Events (SSE) | Low | Low | Most LLM apps. Unidirectional, HTTP-based, works everywhere |
| WebSockets | Low | Medium | Bidirectional needs: voice, real-time collaboration |
| Long Polling | High | Low | Legacy clients that cannot handle SSE or WebSockets |

# Error Handling

Three Layers

1. API Failure. The LLM returns 429 (rate limit), 500 (server error) or times out.
    
    Solution: Exponential back of with jitter starts on 1 second, double each retry, max 3 retries.

```
Attempt 1: immediate
Attempt 2: 1s + random(0, 0.5s)
Attempt 3: 2s + random(0, 1.0s)
Attempt 4: 4s + random(0, 2.0s)
Give up: return fallback response
```

2. Model failures. The model returns malformed JSON, hallucinates a function name or produce an output that fails validation.

    Solution: Retry with a corrected prompt, include the error in  the retry message so the model can self-correct.

3. Application failures. A downstream service is unreachable, vector storage is slow, guardraiul throws an expection.

    Solution: Graceful degradation. IF RAF context is unavailable, proceed without it. If cache is down, bypass it. Never let a secondary system crash the primary flow.
    
| Failure | Retry? | Fallback | User Impact |
|---------|--------|----------|-------------|
| API 429 (rate limit) | Yes, with backoff | Queue the request | "Processing, please wait..." |
| API 500 (server error) | Yes, 3 attempts | Switch to fallback model | Transparent to user |
| API timeout (>30s) | Yes, 1 attempt | Shorter prompt, smaller model | Slightly lower quality |
| Malformed output | Yes, with error context | Return raw text | Minor formatting issues |
| Guardrail block | No | Explain why request was blocked | Clear error message |
| Vector store down | No retry on vector store | Skip RAG context | Lower quality, still functional |
| Cache down | No retry on cache | Direct LLM call | Higher latency, higher cost |

**Fallback model chain.** When your primary model is unavailable, fall through a chain:

```
claude-sonnet-4-20250514 -> gpt-4o -> gpt-4o-mini -> cached response -> "Service temporarily unavailable"
```

# Observability

1. Structured logging - Every request produce a JSON log entry with: request_id, user_id, prompt_template_name, model, input_tokens, output_tokens, latency(ms), cache hit/miss, guardrail pass/fail, cost(USD) and errors.


2. Tracing - A request may touch 5-8 components. OpenTelemetry traces let you see the full journey.
**Metrics dashboard.** The five numbers every LLM team watches:

| Metric | Target | Why |
|--------|--------|-----|
| P50 latency | < 2s | Median user experience |
| P99 latency | < 10s | Tail latency drives churn |
| Cache hit rate | > 30% | Direct cost savings |
| Guardrail block rate | < 5% | Too high = false positives annoying users |
| Cost per request | < $0.01 | Unit economics viability |

# A/B Testing in Production

1. Shadow mode - Run new prompts on 100% of traffic but only log the result -- do not show to users. Compare against current prompt without user risk.

2. Percentage rollout - Route 10% of traffic to new prompt. Monitor metrics, If quality holds, increase to 25%, then 50%, if quality drops, instant rollback

# Scaling

| Scale | Architecture | Infra |
|-------|-------------|-------|
| 0-1K DAU | Single FastAPI server, sync calls | 1 VM, $50/month |
| 1K-10K DAU | Async FastAPI, semantic cache, queue | 2-4 VMs + Redis, $500/month |
| 10K-100K DAU | Horizontal scaling, load balancer, async workers | Kubernetes, $5K/month |
| 100K+ DAU | Multi-region, model routing, dedicated inference | Custom infra, $50K+/month |

Key scaling patterns:

    Async everywhere. Never block a web server thread on an LLM call. Use asyncio and httpx.AsyncClient.
    Queue-based processing. For non-real-time tasks (summarization, analysis), push to a queue (Redis, SQS) and process with workers. Return a job ID, let the client poll.
    Connection pooling. Reuse HTTP connections to LLM providers. Creating a new TLS connection per request adds 100-200ms.
    Horizontal scaling. LLM apps are I/O bound, not CPU bound. A single async server handles 100+ concurrent requests. Scale servers, not cores.


### Cost Projection

Before you ship, estimate your monthly cost. This spreadsheet decides if your business model works.

| Variable | Value | Source |
|----------|-------|--------|
| Daily Active Users (DAU) | 10,000 | Analytics |
| Queries per user per day | 5 | Product analytics |
| Avg input tokens per query | 1,500 | Measured (system + context + user) |
| Avg output tokens per query | 400 | Measured |
| Input price per 1M tokens | $5.00 | OpenAI GPT-5 pricing |
| Output price per 1M tokens | $15.00 | OpenAI GPT-5 pricing |
| Cache hit rate | 35% | Measured from cache metrics |
| Effective daily queries | 32,500 | 50,000 * (1 - 0.35) |

**Monthly LLM cost:**
- Input: 32,500 queries/day x 1,500 tokens x 30 days / 1M x $2.50 = **$3,656**
- Output: 32,500 queries/day x 400 tokens x 30 days / 1M x $10.00 = **$3,900**
- **Total: $7,556/month** (with caching saving ~$4,070/month)

Without caching, the same traffic costs $11,625/month. A 35% cache hit rate saves 35% on LLM costs. This is why Lesson 11 exists.

### The Deployment Checklist

15 items. Ship nothing until every box is checked.

| # | Item | Category |
|---|------|----------|
| 1 | API keys stored in environment variables, not code | Security |
| 2 | Rate limiting per user (10-50 req/min default) | Protection |
| 3 | Input guardrails active (prompt injection, PII) | Safety |
| 4 | Output guardrails active (content filtering, format validation) | Safety |
| 5 | Semantic cache configured and tested | Cost |
| 6 | Streaming enabled for all chat endpoints | UX |
| 7 | Exponential backoff on all LLM API calls | Reliability |
| 8 | Fallback model chain configured | Reliability |
| 9 | Structured logging with request IDs | Observability |
| 10 | Cost tracking per request and per user | Business |
| 11 | Health check endpoint returning dependency status | Ops |
| 12 | Max token limits on input and output | Cost/Safety |
| 13 | Timeout on all external calls (30s default) | Reliability |
| 14 | CORS configured for production domains only | Security |
| 15 | Load test with 100 concurrent users passing | Performance |
