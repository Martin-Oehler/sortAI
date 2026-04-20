# sortAI

Automatically sort PDFs into an existing folder hierarchy using a locally-running LLM. sortAI reads each document, navigates your archive structure, picks the right folder, and generates an appropriate filename — all without moving anything to the cloud.

> **This project is WIP**

## How it works

1. **Extract** — reads the text content of a PDF
2. **Navigate** — traverses your archive folder tree to find the best target
3. **Name** — generates a descriptive filename
4. **Move** — places the file

An LM Studio server provides the local LLM.

## Requirements

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) running locally (for the full pipeline)

## Getting started

```bash
# Clone and enter the repo
git clone https://github.com/Martin-Oehler/sortAI.git
cd sortAI

# Create a virtual environment and install
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Create your config file
cp config/config.example.toml config/config.toml
# Edit config/config.toml with your inbox path, archive path, and LM Studio settings
```

### Configuration

`config/config.toml` (copy from `config/config.example.toml`):

```toml
inbox    = "/path/to/inbox"      # folder to watch for incoming PDFs
archive  = "/path/to/archive"    # root of your existing document archive
log_file = "logs/sortai.jsonl"

[lm_studio]
base_url    = "http://localhost:1234"
model       = "your-model-id"
temperature = 0.2
max_tokens  = 2048
```

## CLI reference

```
sortai [--config FILE] [--dry-run] COMMAND
```

| Command | Description |
|---------|-------------|
| `sortai config` | Print current configuration |
| `sortai extract PDF_FILE [-n MAX_CHARS]` | Extract and display text from a PDF |
| `sortai tree` | Print the archive folder tree |
| `sortai ping` | *(planned)* Test LM Studio connection |
| `sortai process PDF_FILE` | *(planned)* Run the full sort pipeline |
| `sortai watch [--once]` | *(planned)* Watch inbox and auto-process |

**Examples**

```bash
# Preview the first 1000 characters of a document
sortai extract ~/Downloads/invoice.pdf -n 1000

# Show the archive folder structure
sortai tree

# Simulate a sort without moving files
sortai --dry-run process ~/Downloads/invoice.pdf
```

## Running the tests

```bash
pytest tests/
```

Run with verbose output and coverage:

```bash
pytest tests/ -v --cov=src/sortai
```

## License

MIT — see [LICENSE](LICENSE).
