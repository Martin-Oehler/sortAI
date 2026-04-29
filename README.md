# sortAI

Automatically sort PDFs into an existing folder hierarchy using a locally-running LLM. sortAI reads each document, navigates your archive structure, picks the right folder, and generates an appropriate filename — all without moving anything to the cloud.

## How it works

1. **Extract** — reads the text content of a PDF
2. **Navigate** — traverses your archive folder tree to find the best target
3. **Name** — generates a descriptive filename
4. **Move** — places the file

An LM Studio server provides the local LLM.

## Requirements

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) running locally

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

### Setting up LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/).
2. Download a model inside LM Studio (e.g. `google/gemma-4-e4b`).
3. Set the downloaded model's identifier as `model` in `config/config.toml`.
4. Start the local server:
   - Click the **Developer** tab in the left sidebar.
   - Click the toggle next to **Status: Stopped** to start the server.
   - The status should change to **Running** on port `1234`.
5. Verify the connection:
   ```bash
   sortai ping
   ```
   You should see the model load, a short response from the LLM, and a confirmation that the model was unloaded.

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
| `sortai ping` | Test LM Studio connection (load, hello, unload) |
| `sortai process PDF_FILE [--verbose] [--warm]` | Run the full sort pipeline on a single PDF |
| `sortai watch [--once] [--verbose] [--review]` | Watch inbox and auto-process new PDFs |
| `sortai dashboard [--port PORT] [--no-browser]` | Start the review dashboard web server |
| `sortai log [-n N]` | Show recent sort decisions |
| `sortai report` | Regenerate the HTML audit report |
| `sortai validate sample OUTPUT [-n N]` | Sample N PDFs from archive into a test set file |
| `sortai validate run TEST_SET_FILE [--verbose]` | Run the pipeline against a test set and score accuracy |

**Examples**

```bash
# Preview the first 1000 characters of a document
sortai extract ~/Downloads/invoice.pdf -n 1000

# Show the archive folder structure
sortai tree

# Test the LM Studio connection
sortai ping

# Simulate a sort without moving files
sortai --dry-run process ~/Downloads/invoice.pdf

# Create a validation test set (20 randomly sampled PDFs from the archive)
sortai validate sample my_test_set.json -n 20

# Evaluate sorting accuracy against the test set (no files are moved)
sortai validate run my_test_set.json
```

## Review dashboard

The review dashboard lets you inspect LLM decisions before files are moved. It is a persistent local web server that can run at any time, independently of the watcher.

### Workflow

1. **Start the dashboard** (keep it running in one terminal):
   ```bash
   sortai dashboard
   # Opens http://localhost:8765 in your browser automatically
   ```

2. **Watch in review mode** (in another terminal):
   ```bash
   sortai watch --review
   ```
   Instead of moving files directly, the watcher places each incoming PDF in a `_review/` staging folder and adds an entry to `logs/review_queue.json`.

3. **Review in the browser**: staged items appear in the "Needs Review" section at the top. Click a row to preview the PDF. Press **Accept ✓** (or `Y`) to move the file to the proposed archive location, or **Reject ✗** (or `N`) to move it to `_rejected/`.

4. The history log below shows all past decisions (auto-accepted items from normal watch mode as well as accepted/rejected review items), updated live as files are processed.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `J` / `K` | Move focus down / up |
| `Y` | Accept focused staged item |
| `N` | Reject focused staged item |

### Dashboard configuration

```toml
[review]
port = 8765
auto_open_browser = true
# staging_dir  = "/path/to/_review"    # default: inbox parent / "_review"
# rejected_dir = "/path/to/_rejected"  # default: inbox parent / "_rejected"
```

### Normal mode is unchanged

Running `sortai watch` **without** `--review` behaves exactly as before — files are auto-moved and logged immediately.

## Validation

The `validate` commands let you measure how well sortAI sorts documents by comparing its decisions against your existing archive layout.

```bash
# Step 1: sample PDFs already sorted in your archive
sortai validate sample test_set.json -n 50

# Step 2: run the pipeline in dry-run mode and score the results
sortai validate run test_set.json

# Example output:
# ┌── Validation Results ──────────────────────────────────┐
# │  # │ File            │ Ground Truth │ Predicted  │ ✓  │
# │  1 │ statement.pdf   │ bank/2024    │ bank/2024  │ ✓  │
# │  2 │ contract.pdf    │ legal        │ legal/misc │ ~  │
# └────────────────────────────────────────────────────────┘
# Accuracy: 42/50 (84.0%) exact  |  47/50 (94.0%) partial  |  0 error(s)
```

The test set is a plain JSON file — commit it to track accuracy changes over time, or use it to compare different models and prompt configurations.

## Giving the LLM more context

By default the LLM sees only the names of available sub-folders at each navigation step. Two optional mechanisms provide richer context:

### Subfolder previews

Each listed folder automatically shows a sample of its own sub-folders, helping the LLM understand the folder's internal structure:

```
- invoices (contains: 2024, 2023, 2022)
- contracts (contains: active, archived)
- bank-statements
```

Control the number of sub-folders shown with `subfolder_preview_count` (default: 5, set to 0 to disable).

### Folder descriptions

Place a `folder-description.md` file inside any folder to give the LLM a plain-language description of what belongs there:

```
# archive/invoices/folder-description.md
Supplier, utility, and service invoices sorted by calendar year.
```

The description is appended to the folder's entry in the navigation prompt:

```
- invoices (contains: 2024, 2023, 2022) — Supplier, utility, and service invoices sorted by calendar year.
```

Both the filename and the subfolder preview count are configurable in `config/config.toml`:

```toml
folder_description_filename = "folder-description.md"
subfolder_preview_count = 5
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
