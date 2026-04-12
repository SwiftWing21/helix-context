# Enterprise — Federated Genetic Access Gates

> *"We may actually be forerunners in merging nature with silicon data handling."*
> — Max, 2026-04-12

This document argues that for organizations with more than ~5 engineers,
helix-context doesn't just make retrieval *better* — it becomes
**architecturally necessary**. The token economics, security requirements,
and compliance burden at team scale all push toward a retrieval substrate
that conventional MCP + RAG tooling can't provide.

It also makes a larger claim: the enterprise problems helix solves
(multi-tenant isolation, audit trails, access control, semantic
deduplication, cache efficiency) are problems biology solved 3.5 billion
years ago. Porting those solutions from carbon to silicon is what the
architecture does.

---

## The MCP / helix boundary

### What MCP does that helix doesn't replace

- **Write operations** — send email, create ticket, edit file, run migration
- **Real-time data** — stock prices, current time, live sensor readings
- **Actions with side effects** — anything that changes state outside helix
- **Stateless tool invocation** — fire-and-forget commands

These remain MCP's domain. Helix is read-side infrastructure.

### What MCP currently does that helix replaces

Empirically, most MCP servers in production are retrieval proxies — they
fetch data from a source, format it, return it as a tool response, and the
LLM reads it:

- `search_slack` → returns matching messages
- `query_wiki` → returns matching pages
- `lookup_customer` → returns customer record
- `search_docs` → returns relevant documentation
- `find_code` → returns matching code (this IS helix, re-implemented badly)
- `get_past_incidents` → returns similar past tickets

All of these collapse into a single architectural primitive when the data
lives in a genome:

```
MCP server (current)              Helix gene class (replacement)
─────────────────────────────────────────────────────────────────
search_slack tool            →    slack_message genes + retrieval
query_wiki tool              →    wiki_page genes + retrieval
lookup_customer tool         →    customer_record genes + retrieval
search_docs tool             →    doc genes + retrieval
find_code tool               →    code genes + retrieval (this is SIKE)
get_past_incidents tool      →    incident_report genes + retrieval
```

**The 70% rule** (rough empirical estimate): about 70% of MCP servers in
enterprise production are doing retrieval, not actions. That 70% becomes
helix gene classes. The remaining 30% (writes, real-time, actions) stays MCP.

### Why helix wins at retrieval specifically

| Concern | MCP retrieval server | Helix gene class |
|---|---|---|
| **Authorization** | Ad-hoc per server, usually trust-client | Native via `gene_attribution.party_id` |
| **Caching** | Per-server, no cross-tool dedup | Substrate-level, semantic dedup via gene_id |
| **Semantic search** | Usually keyword or naive vector | 13 dimensions (FTS5, SPLADE, SEMA, cymatics, etc.) |
| **Compression** | None (full payloads returned) | Kompress to ~10% of raw size |
| **Audit trail** | Per-server logs (if any) | gene_attribution is an audit log by design |
| **Cross-source queries** | Requires agent orchestration | Single query across all gene classes |
| **Access revocation** | Per-server implementation | Single participant_id flush |
| **Multi-tenant isolation** | Per-server implementation | Native via party_id + cache lanes |

## The federated access model (already half-built)

Helix already has the data model for organizational deployment. What's
missing is the edge layer that enforces it.

### Schema (already shipped)

```python
# helix_context/schemas.py
class Party:
    party_id: str          # organization / tenant / principal
    name: str
    created_at: datetime

class Participant:
    participant_id: str
    party_id: str          # which org does this user belong to
    role: str              # "dev", "analyst", "support", "auditor"
    created_at: datetime

class GeneAttribution:
    gene_id: str
    party_id: str          # the org that owns this gene
    participant_id: str    # the user/agent who contributed it
    authored_at: datetime
```

### Query-side enforcement (already shipped in `d2e0219`)

```python
# helix_context/genome.py — already in production
def query_genes(self, query_terms, party_id=None, ...):
    """
    When party_id is provided, Tiers 1/2/3 all get:
      AND g.gene_id IN (
          SELECT gene_id FROM gene_attribution WHERE party_id = ?
      )
    Attributed genes also get a +0.5 scoring bonus.
    """
```

The bones of federated access are in place. A query with `party_id=X`
can only return genes attributed to X. Cross-tenant leakage is prevented
at the SQL layer, not the application layer.

### What's missing (the edge layer)

Four components need to be built for enterprise deployment:

**1. Session / auth layer**
```
POST /auth
  Authorization: Bearer <oauth_token>
  → { session_token, participant_id, party_id, role, expires_at }

Subsequent calls:
  GET /context?query=...
  Authorization: Bearer <session_token>
  → edge resolves token → participant → party → role → filtered query
```

**2. Role → gene-class access matrix**
```yaml
# Example admin config
roles:
  dev:
    can_access_sources: ["code", "docs", "slack:engineering"]
    can_access_gene_classes: ["constitutive", "inducible"]
    rate_limit_per_day: 50_000
    max_genes_per_expression: 12

  analyst:
    can_access_sources: ["data_warehouse", "docs", "slack:analytics"]
    can_access_gene_classes: ["constitutive"]
    rate_limit_per_day: 10_000

  customer_support:
    can_access_sources: ["customer_records", "docs", "tickets"]
    cannot_access_sources: ["code", "engineering/*"]
    rate_limit_per_day: 20_000

  auditor:
    can_access_sources: ["*"]
    cannot_write: true
    audit_log: verbose
```

**3. Per-party cache isolation**

Cache lane keyed by `(party_id, participant_id)` — tenant A's queries
never contaminate tenant B's cache, even if they happen to hash the same
prompt prefix. This is the security-critical bit that most shared-account
SaaS architectures get wrong.

**4. Admin UI**

Configure roles, grant access, revoke participants, audit query logs.
Web dashboard backed by the existing launcher infrastructure (per
`helix_context/launcher/`). Not hard — just hasn't been prioritized.

## Session and auth semantics

The right pattern splits state between durable (genome) and ephemeral
(session):

```
Session start:
  → Client presents oauth/SAML token
  → Edge resolves to participant_id + party_id + role
  → TCM session context initialized (per-participant, NOT global)
  → Cache lane attached: (party_id, participant_id)
  → Allowed gene classes loaded from role config

During session:
  → Every query enforces party_id filter
  → Retrieval scoped to role's allowed gene classes
  → Per-session TCM drift (one participant's history doesn't bleed)
  → Rate limits enforced per-participant

Session lapse / token drop:
  → Cache lane persists (keyed by party+participant, not by session token)
  → Reconnection picks up where left off with fresh TCM context
  → Co-activation history durable at genome level

Admin revokes participant:
  → Token invalidated at edge
  → Cache lane flushed
  → participant_id retains existing gene_attribution (audit trail preserved)
  → Cannot create new genes or query existing ones
```

**The biological parallel**: your genome is stable day-to-day. Your
working memory resets when you sleep. Your identity (party/participant)
is immutable. Your permissions (role) can change. Helix's session model
matches exactly this split.

## Enterprise economics — why it's NECESSARY, not optional

From [`ECONOMICS.md`](ECONOMICS.md): solo operator at 98.3% cache hit
rate captures an 87x cash-to-compute arbitrage. At team scale, this
effect is **superlinear**, not linear.

### The sharing multiplier

Solo usage pattern:
- Each query pays cache-write once
- Subsequent identical queries pay cache-read (10x cheaper)
- Cache hit rate: 60-99% depending on hygiene

Team usage pattern without helix:
- Dev A asks "how does auth work?" — pays cache-write
- Dev B asks the same question 10 minutes later — ALSO pays cache-write
  (different session, different prompt cache, no sharing)
- Dev C asks a semantically-equivalent but textually-different version
  — ALSO pays cache-write
- Effective cache hit rate: 20-40% (same-user only)

Team usage pattern WITH helix:
- First query ingests relevant content as genes
- Genes are now in the genome, retrievable at substrate cost
- Dev A's query: hits cache + retrieves genes
- Dev B's similar query: hits SAME cache + retrieves SAME genes
- Dev C's semantic variant: hits SAME cache + retrieves SAME genes
  (because SPLADE/cymatics dedup at the semantic level)
- Effective cache hit rate at team level: 70-90%

### Cost example (10-engineer team)

**Without helix (naive RAG + MCP retrieval tools):**
```
Per engineer: ~10M tokens/day effective API consumption
  × 10 engineers                          = 100M tokens/day
  × 250 working days                      = 25B tokens/year
  × blended $1.50/M (assuming some cache) = $37,500/year
  × retail multiplier (no sub arbitrage)  = $150,000-$300,000/year
```

**With helix + shared genome:**
```
Per engineer: ~2M tokens/day effective (80% served from shared cache)
  × 10 engineers                          = 20M tokens/day
  × 250 working days                      = 5B tokens/year
  × blended $0.50/M (high cache hit)      = $2,500/year
  × per-engineer subscription @ $200/mo   = $24,000/year
  TOTAL                                   = ~$26,500/year
```

**Savings at 10-engineer scale: $125k-$275k/year.** These savings scale
superlinearly with team size — at 50 engineers, the shared-cache effect
becomes dominant and the per-engineer cost approaches the cost of a
single heavy user.

### Security & compliance (the "not optional" part)

For any organization subject to SOC 2, ISO 27001, HIPAA, or GDPR:

| Requirement | MCP retrieval servers | Helix federated access |
|---|---|---|
| Access controls per user | Per-server, often missing | Native via party/participant/role |
| Audit trail of queries | Per-server logs, fragmented | Centralized via gene_attribution + query log |
| Data residency | Depends on per-server impl | Per-party database partitioning |
| Right-to-be-forgotten | Requires per-server purge | Delete by party_id, cascade |
| Cross-tenant isolation | Application-layer enforcement | Substrate-layer enforcement |
| Encryption at rest | Per-server config | Single SQLite with SQLCipher option |

At minimum viable enterprise scale (any company with paying customers +
employee data), the MCP retrieval pattern creates compliance risk that
the helix pattern eliminates natively.

## The biological analog — why this works

Every enterprise problem listed above, biology solved 3.5 billion years ago:

| Enterprise problem | Biological solution |
|---|---|
| Multi-tenant isolation | Cell membranes + tissue boundaries |
| Attribution (who made this?) | Every protein traceable to its gene |
| Access control | Receptor-ligand specificity + MHC presentation |
| Audit trail | Transcriptomics — every expression event is recorded |
| Rate limiting | Feedback inhibition + chromatin states |
| Semantic deduplication | Alternative splicing — same gene, many products |
| Cache efficiency | Protein reuse + selective degradation |
| Role-based permissions | Cell differentiation + tissue-specific expression |
| Session state vs persistent state | Working memory (hippocampus) vs long-term memory (cortex) |

The helix architecture is a thin silicon layer over these biological
primitives. The reason it handles enterprise requirements natively is
that **enterprise requirements are restatements of multi-cellular
organizational problems.** A company with 1,000 employees and 10
customer tenants has the same information-architecture challenges as a
human body with 37 trillion cells and a functioning immune system.

## The forerunners claim

Here's the honest framing of what helix represents:

1. **Biology had 3.5 billion years** to evolve multi-cellular information
   handling — isolation, attribution, access control, compression,
   selective expression, audit.

2. **Enterprise software has had ~60 years** to evolve multi-tenant
   information handling — and has done it poorly. Most architectures are
   one of:
   - Row-level security glued onto a relational DB (leaky)
   - Per-tenant databases (doesn't scale)
   - Application-layer enforcement (single bug = breach)

3. **Nobody (to current public knowledge)** has systematically ported
   biology's information-handling primitives into enterprise software.
   People have used biological *metaphors* (neural networks, genetic
   algorithms, evolutionary computation), but those are usually just
   labels on unrelated math.

4. **Helix-context is an attempt to do the actual port** — not metaphor,
   implementation. Genes have DNA-like properties (content-addressable,
   immutable, attributed). Retrieval uses biological signals
   (co-activation, chromatin states, epigenetic decay). Access control
   matches cellular specificity patterns. Cache efficiency follows
   protein reuse economics.

5. **The fact that biological patterns happen to solve enterprise
   problems well is not a coincidence** — both are multi-tenant
   information systems with strict access requirements and high cost of
   errors. Biology had longer to optimize.

**Forerunners** is a strong claim, but the architecture supports it. When
other vendors eventually build "federated AI retrieval for the enterprise,"
they will re-derive helix's schema because it's the schema biology
arrived at. The question is whether they'll know that's what they're
doing.

## What's built vs what's next

### Built (ready for enterprise pilot)
- ✅ Gene + Party + Participant + Attribution schema
- ✅ `party_id` filter in query_genes (scoping)
- ✅ Attribution bonus in retrieval scoring
- ✅ 13-dim retrieval with 98%+ cache efficiency
- ✅ Kompress semantic compression
- ✅ Session registry + TCM temporal context
- ✅ HITL event logging (audit primitive)
- ✅ Density gate + chromatin tiers (lifecycle management)
- ✅ Launcher infrastructure (could host admin UI)

### Not built (needed for enterprise production)
- ⬜ OAuth / SAML edge auth layer
- ⬜ Role → gene-class access matrix
- ⬜ Per-party cache isolation at process level
- ⬜ Admin UI for role management + audit review
- ⬜ Data residency options (per-party DB partitioning)
- ⬜ Rate limiting per participant
- ⬜ Encryption at rest (SQLCipher integration)
- ⬜ Per-party billing metering
- ⬜ Ingest-time PII detection / redaction
- ⬜ Compliance documentation (SOC 2 readiness, ISO 27001 mapping)

None of the "not built" items are research-grade. They're all
implementation work with well-understood patterns from enterprise
software. The research work is already done — in biology, over 3.5
billion years.

## Who this document is for

- **Operators** evaluating helix for organizational deployment
- **Architects** comparing helix to MCP/RAG alternatives
- **Future-helix-devs** planning the auth/admin roadmap
- **Compliance teams** mapping helix primitives to regulatory requirements
- **Investors / partners** understanding the enterprise positioning

It is NOT a sales pitch. It's a design document that happens to make the
enterprise case inevitable once you understand what's been built.

## Related docs

- [`MISSION.md`](MISSION.md) — the biological-substrate philosophy
- [`ECONOMICS.md`](ECONOMICS.md) — cost arbitrage at solo scale
- [`BENCHMARKS.md`](BENCHMARKS.md) — retrieval quality methodology
- [`DIMENSIONS.md`](DIMENSIONS.md) — 13-dimension retrieval primitives
- [`FUTURE/LANGUAGE_AT_THE_EDGES.md`](FUTURE/LANGUAGE_AT_THE_EDGES.md) —
  design endpoint for math-only substrate

---

**Bottom line**: for solo operators, helix is a cost optimization. For
enterprises, helix is a compliance primitive. The same architecture
serves both because biology had the same problem and solved it once.
