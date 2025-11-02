# Super Brain

**Super Brain** is a modular, local-first toolkit for querying, analyzing, and automating reasoning over your own files and data sources. It emphasizes **simple setup**, **clean structure**, and **reproducible runs** so you can start quickly and extend safely.

---

## ✨ Key Features

- **Local-first**: Run entirely on your machine; keep full control of data and artifacts.
- **Modular pipelines**: Swap ingestion, processing, indexing, and query components without rewrites.
- **Config-driven**: Choose sources, parameters, and modes via a single config file.
- **Semantic retrieval (optional)**: Build lightweight indexes for fast, relevant lookups.
- **Scriptable CLI**: Automate end-to-end tasks and batch jobs.
- **Reproducible**: Deterministic settings, logs, and saved outputs.

---

## 📁 Suggested Project Structure

> Actual layout may differ. Use this as a quick map.

super-brain/
├─ src/
│ ├─ core/ # shared utils & abstractions
│ ├─ ingestion/ # CSV/JSON/filesystem/SQL readers
│ ├─ processing/ # cleaning, chunking, embeddings
│ ├─ retrieval/ # indexing & query strategies
│ ├─ cli/ # CLI entry points
│ └─ app/ # optional web/UI hooks
├─ configs/ # YAML/JSON config files
├─ data/ # local data (typically gitignored)
├─ outputs/ # logs, indexes, run artifacts
├─ tests/ # unit/integration tests
├─ requirements.txt # Python deps (if Python stack)
├─ package.json # Node deps (if Node stack)
├─ .env.example # sample environment variables
└─ README.md


---

## 🧰 Prerequisites

Install what matches your stack:

- **Git**
- **Python ≥ 3.10** (if using Python toolchain)
  - `pip` (or `uv`/`pipx`)
- **Node.js ≥ 18** (if using Node toolchain)
  - `npm` or `pnpm` or `yarn`

If optional integrations need secrets/keys, copy the example env file:

```bash
cp .env.example .env
# then edit .env with required values for optional features

🚀 Quick Start
Option A — Python workflow
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
. .\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run a sample pipeline
python -m src.cli.run --config configs/quickstart.yaml

Option B — Node workflow
# 1) Install dependencies
npm install
# or: pnpm install / yarn

# 2) Run a sample task
npm run start -- --config configs/quickstart.json


Use whichever stack this repository actually contains. If both exist, start with one path and extend gradually.

🧪 Common CLI Usage
# Ingest and index a folder of Markdown files
superbrain ingest --source ./data/notes --type markdown --index local

# Ask a question over indexed data
superbrain query --ask "What are the key topics?" --topk 5

# Run an end-to-end pipeline from a config
superbrain run --config ./configs/quickstart.yaml


If superbrain isn’t installed as a global command, call the module/script directly:

python -m src.cli.run --config ./configs/quickstart.yaml
# or
node ./src/cli/run.js --config ./configs/quickstart.json

⚙️ Configuration

Keep runtime options in configs/*:

paths: input/output directories

pipeline: enable/disable stages (ingest → process → index → query)

params: chunk sizes, embedding model, retrieval strategy, top-k, etc.

Example (YAML):

paths:
  input: "./data"
  outputs: "./outputs"

pipeline:
  ingest: true
  process: true
  index: true
  query: false

params:
  chunk_size: 800
  overlap: 100
  embedding_model: "local-mini"
  retrieval: "bm25"
  topk: 5

🧱 Examples

Index local notes

superbrain ingest --source ./data/notes --type markdown --index local


Query for an answer

superbrain query --ask "Summarize meeting actions from last week" --topk 3


Batch run via config

superbrain run --config ./configs/batch.yaml

🧰 Development Tips

Keep small sample files in data/ for fast iteration.

Prefer pure, testable functions and typed signatures.

Add tests for every module/bugfix:

pytest -q       # Python
# or
npm test        # Node


Log parameters and decisions to outputs/ for reproducibility.

🐛 Troubleshooting

Command not found → Activate your Python venv or reinstall Node deps.

Missing env values → Copy .env.example to .env and fill only the needed keys.

Slow queries → Reduce topk, simplify embeddings, or prune large files.

High memory → Process in batches; ensure indexes are on disk rather than RAM.

🗺️ Roadmap (suggested)

Pluggable UI for interactive exploration

Additional connectors (cloud docs/DBs) via adapter pattern

Reranking and multi-vector retrieval

On-disk caches for repeat questions and offline mode

🔏 License

Add your license (e.g., MIT/Apache-2.0) in LICENSE. If none is provided, all rights are reserved by default.
