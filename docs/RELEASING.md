# Releasing Cymatix Context

This is the runbook for the beta-branch flow adopted on 2026-09-01, after
v0.9.1 shipped from `master`. It covers the branch model, what CI runs where,
the receipt gate, pre-releases, the final release, and the post-release
sync, with the exact commands for each step. `scripts/release.py` does the
file edits (version bump, CHANGELOG roll, tag message); it never runs
`git commit`, `git tag` or `git push` itself, so every step that touches the
remote is one you type.

## 1. Branch model

| Branch | Holds | Who merges into it | Notes |
|---|---|---|---|
| `master` | Released code and release tags only | The owner, by PR from `release/**` or `hotfix/**` | `master` always equals the latest tag plus nothing. `git describe --tags master` names the current release. |
| `beta` | Integration: everything headed for the next release | Feature and fix PRs (the default PR base) | Fast-forwarded to each release afterwards. Installable from PyPI as a pre-release when a `vX.Y.ZbN` tag is cut from it. |
| `release/vX.Y.Z` | The release cut: one commit on top of `beta` with the version bump and CHANGELOG roll | Nobody else; it exists to open the PR to `master` | Cut from `beta` once the witness ladder passes. Deleted after the tag. |
| `hotfix/*` | A fix for the released version that cannot wait for the next release | The owner, by PR to `master` | Cut from `master`. After it merges and ships as X.Y.Z+1, merge `master` back into `beta` so the fix is not lost. |

Rules that fall out of the table:

- Feature work never targets `master`. A PR against `master` whose head is not
  `release/**` or `hotfix/**` is wrong and gets retargeted to `beta`.
- `release/**` and `hotfix/**` branches are the only ones that ever change
  the version string.
- `beta` is never force-pushed. Integration merges land as merge commits or
  fast-forwards, so `beta` contains every commit that `master` does.
- A hotfix gets its own patch release (`vX.Y.(Z+1)`) through the same
  final-release steps below, with `hotfix/*` standing in for `release/*`.

## 2. What CI runs where

`.github/workflows/ci.yml` runs five test jobs (`test-linux`,
`test-symbol-graph`, `test-full-suite`, `test-macos-mps`, `test-windows`) on:

- every push to `master`, `beta` and `release/**`;
- every pull request whose base is `master` or `beta`.

The same file carries a sixth job, `release-source`, which enforces the
source-branch rule from section 1: it has a job-level
`if: github.event_name == 'pull_request' && github.base_ref == 'master'`, so it
runs on pull requests targeting `master` and is skipped on everything else —
pushes to `beta` and `release/**`, and pull requests targeting `beta`, are
unaffected by it. It passes when the head branch matches `release/*` or
`hotfix/*` and fails otherwise, telling the author to retarget to `beta`. It is
a required check on `master` (section 8).

`.github/workflows/cla.yml` (job `contributor-signup`) runs on every pull
request against any base branch; it has no branch filter on purpose, so PRs
to `beta`, `release/**` and `master` are all gated by the one-time contributor
sign-up described in `CONTRIBUTING.md`.

`.github/workflows/publish.yml` builds and uploads to PyPI when a GitHub
release is published. GitHub fires `release: published` for pre-releases too,
so a release marked "pre-release" on a `vX.Y.ZbN` tag publishes a beta to
PyPI, and a normal release on a `vX.Y.Z` tag publishes the final. It never
runs on a push or a PR.

A `hotfix/*` branch is exercised by its PR to `master` (the `pull_request`
trigger), which is why it is not listed under `push`.

## 3. The receipt gate

Releases are receipt-gated, not calendar-gated. Two gates apply, at two
different points:

1. **Any shipped-default change needs its paired receipt and its
   `docs/benchmarks/BASELINES.md` row before it merges to `beta`.** A knob and
   its default flip are separate PRs: the knob lands default-inert with a test
   pinning byte-identical legacy behaviour, and the flip PR carries the
   receipt path in its CHANGELOG entry (see the
   `entity_autolink_hub_cutoff` flip in the 0.9.2 CHANGELOG section for the
   shape). No receipt, no flip.
2. **A release needs a merged-stack witness ladder on the 947k bed.** After
   the last PR lands on `beta` and before `release/vX.Y.Z` is cut, the
   integrated head is re-measured as a whole, because the per-PR receipts were
   taken against different bases. The 0.9.1 gate receipt is
   `benchmarks/dogfood/receipts/sweep_v091_gate_2026-08-30.json` (three-arm
   sweep, ALL PASS, ledger row `2026-08-30-v091-gate-sweep`). The 0.9.2
   witness is recorded at
   `benchmarks/dogfood/receipts/sweep_v092_witness_947k_2026-09-08.json`
   (BASELINES row `2026-09-08-v092-merged-stack-witness`), measured after
   PRs #446–#449. The earlier September 4 witness remains historical evidence
   for its own pinned stack. The final witness receipt is
   committed to `beta` before the release branch is cut, so the release cut
   carries it.

A pre-release does not need the witness ladder; that is what makes it a
pre-release. It does need every per-PR receipt already in place, because it
ships the same defaults.

## 4. Pre-releases (`vX.Y.ZbN`)

A pre-release is a snapshot of `beta`, versioned `X.Y.ZbN` where `X.Y.Z` is
the final version it previews and `N` counts up from 1. It is tagged on
`beta`, published as a GitHub pre-release, and pushed to PyPI by
`publish.yml`. Users opt in with `pip install --pre cymatix-context`; a plain
`pip install cymatix-context` keeps resolving to the last final release
because pip ignores pre-releases unless asked.

```bash
git fetch origin
git checkout beta && git pull --ff-only origin beta
python scripts/release.py check                 # versions agree, Unreleased has content, tree clean

python scripts/release.py pre --n 1             # dry run: prints the diff and the next commands
python scripts/release.py pre --n 1 --write     # 0.9.1 -> 0.9.2b1 in pyproject + __init__, note under Unreleased
python -m pytest tests/test_version.py -q

git add pyproject.toml cymatix_context/__init__.py CHANGELOG.md
git commit -m "release: v0.9.2b1 pre-release"
git push origin HEAD:refs/heads/beta

python scripts/release.py tag-message --version 0.9.2b1 > tagmsg.txt
python scripts/release.py tag-message --version 0.9.2b1 --body-only > notes.md
git tag -a v0.9.2b1 -F tagmsg.txt
git push origin v0.9.2b1
gh release create v0.9.2b1 --prerelease --title "v0.9.2b1" --notes-file notes.md
```

Notes:

- Two forms of the same text: an annotated tag has no title field, so
  `git tag -F` wants the message that opens with the `v0.9.2b1` line, while a
  GitHub release renders its own title and would show the version twice.
  `--body-only` drops that leading line, which is the form `--notes-file`
  wants. Both `tagmsg.txt` and `notes.md` are scratch files; neither is
  committed.

- `pre` picks the previewed version for you: on a tree at a released
  `X.Y.Z` it moves to `X.Y.(Z+1)`; on a tree already at `X.Y.ZbM` it stays
  on `X.Y.Z`. Pass `--version 0.10.0` to preview a different final. It
  refuses a base equal to the released version, because `0.9.1b1` would sort
  before the already-published `0.9.1`.
- The tag message and release notes come from the current `## Unreleased`
  section, which is what the pre-release contains.
- After the tag, `beta` stays at `X.Y.ZbN`. Later integration PRs merge on
  top; the next `pre --n 2` replaces the suffix. The final release replaces
  it with `X.Y.Z`.
- Watch the "Publish to PyPI" workflow run and confirm
  `pip install --pre cymatix-context==0.9.2b1` resolves before announcing.

## 5. The final release (`vX.Y.Z`)

Preconditions: the witness ladder receipt is on `beta` with its BASELINES
row, every shipped-default change in `## Unreleased` cites its receipt, the
wiki and `CLAUDE.md` describe the new defaults, and CI is green on `beta`.

```bash
git fetch origin
git checkout beta && git pull --ff-only origin beta
python scripts/release.py check

git checkout -b release/v0.9.2
python scripts/release.py final --version 0.9.2            # dry run
python scripts/release.py final --version 0.9.2 --write    # bump + roll "## Unreleased" into "## 0.9.2 (YYYY-MM-DD)"
python -m pytest tests/test_version.py -q

python scripts/release.py tag-message --version 0.9.2 > tagmsg.txt
python scripts/release.py tag-message --version 0.9.2 --body-only > notes.md
git add pyproject.toml cymatix_context/__init__.py CHANGELOG.md
git commit -m "release: v0.9.2"
git push origin HEAD:refs/heads/release/v0.9.2
gh pr create --base master --head release/v0.9.2 --title "release: v0.9.2" --body-file notes.md
```

`tagmsg.txt` keeps the leading `v0.9.2` line for `git tag -F`; `notes.md` is
the same body with that line dropped (`--body-only`), which is what the PR
body and `gh release create --notes-file` want, since both render the title
themselves. Neither file is committed.

Wait for CI on the PR (the `release/**` push and the PR to `master` both
run the full job set) and merge it with a merge commit, so the release
commit itself stays on the `master` line and `git log master` reads as one
merge per release. Then tag the merge commit and publish:

```bash
git fetch origin
git checkout master && git pull --ff-only origin master
git tag -a v0.9.2 -F tagmsg.txt            # annotated tag on the merge commit
git push origin v0.9.2
gh release create v0.9.2 --title "v0.9.2" --notes-file notes.md
```

`publish.yml` fires on the published release, builds the sdist and wheel,
and uploads them with trusted publishing. Watch the run, then confirm
`pip install cymatix-context==0.9.2` resolves.

`final` leaves a fresh empty `## Unreleased` heading above the new section,
so the first post-release PR to `beta` has somewhere to write its entry.

## 6. Post-release

```bash
# 1. Bring beta up to the release (fast-forward when nothing landed on beta
#    during the release window; otherwise a merge commit, never a rebase).
git checkout beta && git pull --ff-only origin beta
git merge --ff-only master || git merge --no-ff master -m "merge master (v0.9.2) back into beta"
git push origin beta
git push origin --delete release/v0.9.2

# 2. Wiki: GitHub wiki and the site render from the same wiki/ directory.
python scripts/sync_github_wiki.py --dry-run
python scripts/sync_github_wiki.py
python scripts/build_wiki_site.py --site-dir F:/Projects/cymatix-context/site
#    then commit and deploy the site checkout the way the site README says.

# 3. Ledger: add a comment on the 0.9.x ledger issue (#377) naming the tag,
#    the gate receipt, and which deferred items moved.
gh issue comment 377 --body-file release-ledger-note.md
```

For a hotfix the same steps apply with `hotfix/<name>` in place of
`release/vX.Y.Z`; the back-merge into `beta` in step 1 is what carries the
fix forward, so it is not optional.

## 7. Checklist

Pre-release:

- [ ] `python scripts/release.py check` passes on `beta`
- [ ] Every shipped-default change in `## Unreleased` cites its receipt and BASELINES row
- [ ] `pre --n N --write`, `tests/test_version.py` green, commit pushed to `beta`
- [ ] Annotated tag `vX.Y.ZbN` pushed, GitHub release created with `--prerelease`
- [ ] "Publish to PyPI" run green, `pip install --pre cymatix-context==X.Y.ZbN` resolves

Final release:

- [ ] Merged-stack witness ladder receipt committed to `beta` with its BASELINES row
- [ ] Wiki, `CLAUDE.md`, `README.md` and `docs/config-reference.md` describe the shipped defaults
- [ ] CI green on `beta`
- [ ] `release/vX.Y.Z` cut, `final --version X.Y.Z --write`, `tests/test_version.py` green
- [ ] PR to `master` green and merged with a merge commit
- [ ] Annotated tag `vX.Y.Z` on the merge commit, GitHub release published from the CHANGELOG section
- [ ] "Publish to PyPI" run green, `pip install cymatix-context==X.Y.Z` resolves
- [ ] `master` merged or fast-forwarded back into `beta`, release branch deleted
- [ ] `sync_github_wiki.py` and `build_wiki_site.py` run, site deployed
- [ ] Ledger issue #377 updated

## 8. Branch protection

Both rule sets below are applied, as of 2026-09-01. They are classic branch
protection, set through the API; the JSON and `gh` commands are kept here as
the record of what is in force and how to reapply it after an edit in the
GitHub UI. Read the live state with
`gh api repos/mbachaud/Cymatix-Context/branches/<branch>/protection`.

In force on both branches: `enforce_admins` false, `strict` false, force
pushes blocked, deletion blocked.

- `beta` requires six checks: `contributor-signup`, `test-linux`,
  `test-symbol-graph`, `test-full-suite`, `test-macos-mps`, `test-windows`.
- `master` requires those six plus `release-source`, and a pull request with
  0 required approving reviews.

### `beta`

Required checks: the five CI test jobs plus `contributor-signup`; no
force-push, no deletion. `enforce_admins` is false so the owner can still push
a locally merged integration head directly (required status checks apply to
direct pushes as well, and a local merge commit has no check runs, so with
`enforce_admins: true` every integration would have to go through a PR).

```bash
cat > beta-protection.json <<'JSON'
{
  "required_status_checks": {
    "strict": false,
    "contexts": [
      "contributor-signup",
      "test-linux",
      "test-symbol-graph",
      "test-full-suite",
      "test-macos-mps",
      "test-windows"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": false
}
JSON
gh api -X PUT repos/mbachaud/Cymatix-Context/branches/beta/protection --input beta-protection.json
```

### `master`

Same checks as `beta`, plus the `release-source` check and a required pull
request (0 approving reviews), so only `release/**` or `hotfix/**` heads can
merge.

GitHub cannot restrict a protected branch by the PR's source branch. Neither
classic branch protection nor repository rulesets have a head-branch
condition: rulesets filter on the target ref, the repository, and actors,
and they add "require pull request", "require status checks", "block force
pushes", "restrict deletions" and bypass lists, but not "only from these
branches". The guard is therefore a status check that fails when the head is
not `release/**` or `hotfix/**`.

That check is the `release-source` job in `.github/workflows/ci.yml`, and it
is one of `master`'s required contexts. Its job-level
`if: github.event_name == 'pull_request' && github.base_ref == 'master'` keeps
it off every other event, so it never runs on a `beta` push or a PR whose base
is `beta`. On a PR to `master` it passes for a `release/*` or `hotfix/*` head
and otherwise fails with "PRs to master must come from release/** or
hotfix/**; retarget this PR to beta".

Classic protection:

```bash
cat > master-protection.json <<'JSON'
{
  "required_status_checks": {
    "strict": false,
    "contexts": [
      "contributor-signup",
      "release-source",
      "test-linux",
      "test-symbol-graph",
      "test-full-suite",
      "test-macos-mps",
      "test-windows"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": false
}
JSON
gh api -X PUT repos/mbachaud/Cymatix-Context/branches/master/protection --input master-protection.json
```

Rulesets alternative — **not applied**; kept as the equivalent if the
repository ever moves off classic protection (one ruleset for `master`; the
source-branch guard is still the `release-source` status check, because
rulesets cannot express it either). The bypass actor is the repository admin
role so the owner keeps an escape hatch:

```bash
cat > master-ruleset.json <<'JSON'
{
  "name": "master: releases and hotfixes only",
  "target": "branch",
  "enforcement": "active",
  "conditions": { "ref_name": { "include": ["refs/heads/master"], "exclude": [] } },
  "bypass_actors": [ { "actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always" } ],
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "pull_request", "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": false,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": false } },
    { "type": "required_status_checks", "parameters": {
        "strict_required_status_checks_policy": false,
        "required_status_checks": [
          { "context": "contributor-signup" },
          { "context": "release-source" },
          { "context": "test-linux" },
          { "context": "test-symbol-graph" },
          { "context": "test-full-suite" },
          { "context": "test-macos-mps" },
          { "context": "test-windows" } ] } }
  ]
}
JSON
gh api -X POST repos/mbachaud/Cymatix-Context/rulesets --input master-ruleset.json
```

If the ruleset is used, remove the classic protection on `master` first so
the two do not stack (`gh api -X DELETE repos/mbachaud/Cymatix-Context/branches/master/protection`),
or keep both and accept that the stricter rule wins.

### Default branch

`beta` has been the repository's default branch since 2026-09-01. New pull
requests land on the integration branch without the contributor having to
change the base, `git clone` and the repository landing page show `beta`, and
GitHub's "compare" and "new branch" defaults point at it.

```bash
gh repo edit mbachaud/Cymatix-Context --default-branch beta
```

What it costs, and how the docs handle it:

- The landing-page README is `beta`'s. The README on `master` stays the
  released doc, and every wiki and docs link that should describe a shipped
  version points at `blob/master/...` explicitly, so those links keep
  describing the release regardless of the default branch. Links to files
  that only exist on the integration branch — this runbook among them — point
  at `blob/beta/...` instead, because `blob/master/` would 404 until the next
  release carries the file across. `beta`'s README may describe defaults that
  are not on PyPI yet; the wiki header line ("This wiki documents 0.9.x") is
  where that is disclosed.
- `pip install git+https://github.com/mbachaud/Cymatix-Context` installs
  `beta`. Anyone who wants the release installs from PyPI or pins
  `@v0.9.2`.
- Issue templates, `CODEOWNERS` and workflow files are read from the default
  branch for some GitHub features (issue forms, in particular). Keep them in
  sync by merging `master` back into `beta` after each release, which the
  post-release steps already do.
- Existing clones are unaffected; `origin/HEAD` only changes on a fresh
  clone or `git remote set-head origin -a`.

The `release-source` check is what catches a contributor who still opens a
pull request against `master`.
