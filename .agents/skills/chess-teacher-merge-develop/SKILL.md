---
name: chess-teacher-merge-develop
description: >-
  Complete feature-branch → develop PRs for chess_teacher: ensure bump label,
  open/reuse PR, watch CI, fix clear failures or report back, then merge.
  Use when the user asks to merge to develop, finish a feature PR, ship to
  develop, or complete the feature→develop flow. Never merges to main.
---

# Chess Teacher: merge feature → develop

Orchestrate opening, CI-babysitting, and merging a PR into **`develop`**.
GitHub Actions already handle CI checks and post-merge CD (version bump +
dev image). This skill does **not** redeploy or touch **`main`**.

## Hard rules

- Base branch is always **`develop`**. Never create or merge a PR to **`main`**.
- Exactly one bump label: `bump:patch` | `bump:minor` | `bump:major`.
- If unsure which bump to use, choose the **smaller** one (`patch` < `minor` < `major`). Default when unclear: **`bump:patch`**.
- Do not edit `.github/workflows/*` just to make CI green.
- Do not re-run CD / deploy scripts after merge — Actions owns that.
- Prefer `gh` for all GitHub operations. Run from repository root.
- **Remote is source of truth for the PR/CI.** Push before trusting checks; sync against `origin/develop`, not a stale local `develop`. No force-push to `main`/`develop`.

## Bump label choice

| Change shape | Label |
|---|---|
| Bugfix, tests, chore, docs, small hardening | `bump:patch` |
| User-visible feature or meaningful capability (clear) | `bump:minor` |
| Breaking API / data / deploy contract (clear) | `bump:major` |
| Doubtful | **smaller** label (usually `bump:patch`) |

If the PR already has exactly one valid bump label, keep it. If multiple or
none, fix to exactly one using the table above.

## Who runs what

- **User:** “merge this to develop” / “finish the PR”.
- **Agent:** follows the workflow below; uses `scripts/pr_status.py` for status.
- **GitHub Actions:** CI on the PR; CD on push to `develop` after merge.

## Workflow

Copy and track:

```
- [ ] 1. Preflight
- [ ] 2. Local vs remote sync
- [ ] 3. Push feature branch
- [ ] 4. Open or reuse PR → develop
- [ ] 5. Ensure exactly one bump:* label
- [ ] 6. CI loop (until green or blocked)
- [ ] 7. Merge into develop (remote)
- [ ] 8. Report PR URL + note that CD runs on develop
```

### 1. Preflight

```bash
git status
git branch --show-current
git remote -v
git fetch origin
python .agents/skills/chess-teacher-merge-develop/scripts/pr_status.py --json status
```

Abort and ask the user if:

- Current branch is `main` or `develop`
- Working tree has unrelated secrets (`.env`, credentials) that would be committed
- User intent was production / `main` (direct them to a manual release; this skill stops)

### 2. Local vs remote (required)

Always reason about **three refs**, never only the local branch tip:

| Ref | Role |
|---|---|
| `HEAD` (local feature branch) | What you can commit on |
| `origin/<feature>` | What the PR / CI actually builds |
| `origin/develop` | Integration base (after `git fetch origin`) |

Rules:

- **CI and the PR follow the remote feature branch.** After every local commit intended for the PR, `git push` (set upstream once with `git push -u origin HEAD`). Do not assume local-only commits are on the PR.
- **Sync against `origin/develop`**, not a stale local `develop`. Update local `develop` only if needed for convenience; never merge into `develop` locally as a substitute for the GitHub PR.
- Before opening/updating the PR, check divergence:

```bash
git fetch origin
git status -sb
git rev-list --left-right --count origin/develop...HEAD
git rev-list --left-right --count origin/$(git branch --show-current)...HEAD 2>nul || true
```

  - Local **ahead** of `origin/<feature>` → push before relying on CI/PR status.
  - Local **behind** `origin/<feature>` → `git pull` (ff-only if possible); if histories diverged unexpectedly, stop and ask (do not force-pull).
  - Feature branch **behind** `origin/develop` → merge `origin/develop` into the feature branch (prefer merge unless the branch already uses rebase). Resolve conflicts; if intents conflict, stop and ask. Then push.
- **Uncommitted work:** do not commit unless the user already asked to commit or clearly wants that work in the PR. Do not discard local changes.
- **Never** `git push --force` (or `--force-with-lease`) to `main` / `develop`. Avoid force-push to the feature branch unless the user explicitly asks; prefer normal pushes.
- After a successful GitHub merge, **local `develop` is stale** until `git fetch origin` (and optional `git checkout develop && git pull`). Do not treat local `develop` as post-merge truth.

### 3. Push / sync

- Commit only if the user already asked for commits or explicitly wants uncommitted work included.
- Ensure `origin/<feature>` matches `HEAD` for everything that should be in the PR (`git push -u origin HEAD` if needed).
- If the feature branch is behind `origin/develop`, merge `origin/develop` in, then push (see table above).

### 4. Open or reuse PR

If no open PR to `develop` exists for this **remote** head branch:

```bash
gh pr create --base develop --title "..." --body "$(cat <<'EOF'
## Summary
- <1–3 bullets>

## Test plan
- [ ] CI green on this PR

EOF
)"
```

If one exists, reuse it (`gh pr view` / `pr_status.py`). Confirm `base` is `develop` and `head` matches the branch you pushed.

### 5. Bump label

```bash
# Example: set patch (adjust after applying the bump table)
gh pr edit <N> --add-label "bump:patch"
# Remove extras if needed:
# gh pr edit <N> --remove-label "bump:minor"
```

CI job `check-bump-label` fails without exactly one bump label.

### 6. CI loop (required)

Repeat until success or you must report back:

1. Refresh status (**after** remote is up to date with any local fixes):

```bash
python .agents/skills/chess-teacher-merge-develop/scripts/pr_status.py --json status
# or: gh pr checks <N>
```

2. If checks are **pending**, wait and re-poll (sleep / `gh pr checks --watch` with a timeout). Do not merge yet.

3. If checks are **green** and the PR is mergeable → go to step 7.

4. If checks **failed**:

   - Pull failing job logs (`gh run view <id> --log-failed` or the check’s details URL).
   - **Fix only when the solution is clear** (obvious test/lint/type error caused by this PR; missing bump label; branch behind `origin/develop` with an unrelated fixed failure).
   - Commit (if allowed) → **push to `origin/<feature>`** → return to (1). Local-only fixes do not re-run CI.
   - Cap automatic fix attempts at **3** distinct push cycles. If still red → **report back** (do not merge).

5. **Report back** (do not keep guessing) when:

   - Root cause is unclear after reading the failure
   - Fix would require changing CI workflows, unrelated refactors, or prod/main behavior
   - Failures look flaky/infra and are not clearly addressed by merging `origin/develop`
   - Merge conflicts need product intent the agent cannot decide
   - Local and `origin/<feature>` have diverged in a surprising way

   Report: PR URL, failing check names, short cause hypothesis, what you already tried.

### 7. Merge

Only when CI is green on the **remote** PR head, bump label is correct, and the PR is mergeable:

```bash
gh pr merge <N> --merge
```

Prefer `--merge` (merge commit) unless the repo/user already standardized on squash. Do not `--admin` bypass failing checks.

### 8. Done

Return:

- PR URL
- Bump label used
- Confirmation merged into `develop` on the remote
- Reminder: CD on `develop` bumps version and publishes the **development** image (not production)
- Note that local `develop` may need `git fetch` / pull if the user keeps working locally

## Status script

```bash
python .agents/skills/chess-teacher-merge-develop/scripts/pr_status.py --json status
python .agents/skills/chess-teacher-merge-develop/scripts/pr_status.py --json status --pr 123
```

Emits JSON: PR number/url/base/head, bump labels, mergeability, and check rollup. Use it at preflight and inside the CI loop.

## Out of scope

- Merging to `main` / production deploy
- Choosing bump labels for the user without applying the table (always pick one)
- Rewriting CD or manually tagging releases
