# Contributing to Cymatix Context

## Issues — open to everyone

Bug reports and feature requests are welcome from anyone, no sign-up needed.
Please use the issue forms (they capture the surface, OS, model setup, and
installed version we need to reproduce):

- [Bug report](https://github.com/mbachaud/Cymatix-Context/issues/new?template=bug_report.yml)
- [Feature request / enhancement](https://github.com/mbachaud/Cymatix-Context/issues/new?template=feature_request.yml)

## Pull requests — one-time contributor sign-up

PRs are gated on a one-time sign-up (CLA Assistant Lite). The flow:

1. Read the contributor guidelines at <https://cymatixcontext.com/contributing>.
2. Open your PR. The **Contributor sign-up** check will fail with a comment.
3. Post this exact sentence as a comment on your PR:

   > I have read the Contributor Guidelines and I hereby sign up as a Cymatix Context contributor

4. The check turns green (comment `recheck` if it needs a nudge) and applies to
   all your future PRs. Signatures are recorded on the `cla-signatures` branch.

Unsigned PRs stay blocked from merging. Maintainers are allowlisted.

## Branches and releases

- Open PRs against **`beta`**, the integration branch. It is the repository's
  default branch, so a new PR already targets it. `master` holds released code
  and tags only; the only PRs that target it are `release/vX.Y.Z` cuts and
  `hotfix/*` branches, and a `release-source` check rejects anything else.
- A default flip needs its receipt and `docs/benchmarks/BASELINES.md` row in
  the same PR, and the knob ships default-inert in a separate PR first.
- Pre-releases (`vX.Y.ZbN`, `pip install --pre cymatix-context`) are cut from
  `beta`; final releases go through `release/vX.Y.Z` to `master`.

The full runbook, including `scripts/release.py`, is
[`docs/RELEASING.md`](docs/RELEASING.md).

## Development setup

Use Python 3.11+ in a virtual environment; CI uses Python 3.12. Install the
contributor extra for pytest, wiki rendering, and the offline BM25 baseline:

```text
python -m pip install -e ".[dev]"
python -m pytest tests/ -m "not live" -q -rs
```

The contributor extra also includes `httpx2`, the
[HTTP client used by Starlette's TestClient](https://starlette.dev/testclient/).

For the optional coverage exercised by the full Linux CI job, install its
feature extras and the separate spaCy pipeline:

```text
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -e ".[dev,ast,mcp,nli]" psutil platformdirs py-cpuinfo joblib scikit-learn
python -m spacy download en_core_web_sm
python -m pip check
python -m pytest tests/ -m "not live" -q -rs
```

The MCP extra requires SDK 2.x. An older environment can pass `pip check`
while its optional MCP tests skip; reinstall the requested extras when
updating the checkout. The `-rs` output distinguishes absent feature
dependencies and model pipelines from hardware or platform skips. See
[CI](.github/workflows/ci.yml) for the platform-specific jobs and
[setup](docs/SETUP.md) for additional feature extras.

Live tests require Ollama with a model:

```text
python -m pytest tests/ -m live -v -s
```

## Ground rules

- **Receipts culture.** Performance or quality claims need measurements; default
  flips on the retrieval path need benchmark receipts (see
  `docs/benchmarks/BASELINES.md`).
- **Software lexicon.** Biology terms (gene, ribosome, splice) have canonical
  software equivalents in `docs/ROSETTA.md` — new code uses the software terms.
- **Windows counts.** The primary dev bed is Windows; keep subprocess and path
  handling cross-platform.
- **Tests ship with changes.** The suite (~4k tests) must pass without external
  services: `python -m pytest tests/ -m "not live"`.
