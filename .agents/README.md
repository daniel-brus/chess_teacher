# Agent configuration

Version-controlled Cursor agent skills live under [`.agents/skills/`](skills/).

Planning docs for agents:

| Doc | Topic |
|-----|--------|
| [ml-training-roadmap.md](docs/ml-training-roadmap.md) | Baseline + user-bot ML phases (eval splits, ops siblings, notebook, personalization) |

| Skill | Invoke |
|-------|--------|
| [chess-teacher-db](skills/chess-teacher-db/) | `/chess-teacher-db` |
| [import-dag](skills/import-dag/) | `/import-dag` (also auto when moving modules / fixing imports) |

Add new skills as `.agents/skills/<name>/SKILL.md`. Cursor discovers them automatically (reload window after adding).

Note: a local always-apply copy also lives at `.cursor/rules/import-dag.mdc` (`.cursor/` is gitignored). Prefer editing the skill here so the rule stays version-controlled.
