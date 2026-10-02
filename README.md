# concept-embeddings-RAG-Edge

A tournament of retrieval strategies for questions whose evidence is spread over several passages.

> For a question over an arbitrary text corpus, which retrieval strategy recovers the full evidence most often **at a given cost**,
> and does it keep that advantage on a corpus nobody looked at while choosing it?

Every candidate is measured against the literature's best recipes ("ghosts"), reproduced on the same harness, with its offline and online cost beside its score.
Strategies are chosen on open corpora (the terrain) and tested once on a locked corpus (the exam).

## Status

Starting.
The plan is in [docs/plans/0_plan_maestro.md](docs/plans/0_plan_maestro.md); Phase 00 maps the state of the art before anything is measured.

## Where it comes from

This project succeeds `concept-embeddings-RAG`, a closed research line on the minimum structural signal dense retrieval needs for multi-hop evidence.
Its results, the decisions behind this project and the hand-over state are in [docs/refs/](docs/refs/):
[research summary](docs/refs/research_summary.md), [charter](docs/refs/successor_project_charter.md), [hand-over](docs/refs/successor_project_handover.md).

## How the work is run

Spec-driven phases with frozen criteria, autonomous implementation and adversarial review (SDD Lite).
The working guide, in Spanish, is [docs/como-trabajar.md](docs/como-trabajar.md); agent rules are in [AGENTS.md](AGENTS.md).
`npm run check` runs the repository checks; `npm ci --ignore-scripts --no-audit --no-fund` installs the tooling.

## Old data

The harness reads the old project's `data/` in place, read-only: default `../concept-embeddings-RAG/data` beside this repository, override with the env var `OLD_DATA_ROOT`.

## Author

Mikel Ugarte-Gil, independent researcher.
