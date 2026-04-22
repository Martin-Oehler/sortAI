"""File move operations and decision logging."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape as _esc
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def move_file(src: Path, dest_dir: Path, new_name: str, dry_run: bool) -> Path:
    """Move *src* into *dest_dir* with name *new_name*.

    Creates *dest_dir* if needed; appends _2, _3 … on collision.
    Returns the final destination path (even in dry_run mode).
    Does NOT move anything when dry_run=True.
    """
    stem = Path(new_name).stem
    suffix = Path(new_name).suffix or ".pdf"
    dest = dest_dir / new_name
    counter = 2
    while dest.exists():
        dest = dest_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
    return dest


def log_decision(
    src: Path,
    dest: Path,
    summary: str,
    dry_run: bool,
    log_path: Path,
) -> None:
    """Append a JSON-lines entry to *log_path* and regenerate the HTML report."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "original_path": str(src.resolve()),
        "new_path": str(dest.resolve()),
        "summary": summary,
        "dry_run": dry_run,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    render_html_report(log_path)


def render_html_report(log_path: Path) -> None:
    """Read all JSONL entries from *log_path* and write a self-contained HTML report."""
    entries: list[dict] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    html_path = _html_path(log_path)
    html_path.write_text(_build_html(entries), encoding="utf-8")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _html_path(log_path: Path) -> Path:
    return log_path.with_name(log_path.stem + "_report.html")


def _build_html(entries: list[dict]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(entries)
    dry_run_count = sum(1 for e in entries if e.get("dry_run"))
    rows = _build_rows(entries)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>sortAI Audit Log</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1600px; margin: auto; padding: 1rem; }}
    h1 {{ margin-bottom: 0.25rem; }}
    .meta {{ color: #555; margin-bottom: 0.75rem; }}
    #filterInput {{ width: 100%; padding: 0.5rem; margin-bottom: 1rem; font-size: 1rem; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ cursor: pointer; background: #2c3e50; color: #fff; padding: 0.5rem 0.6rem; text-align: left; user-select: none; }}
    th:hover {{ background: #34495e; }}
    td {{ padding: 0.4rem 0.6rem; vertical-align: top; font-size: 0.85rem; border-bottom: 1px solid #e0e0e0; }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    tr.dry-run {{ background: #fff3cd; }}
    tr.dry-run:nth-child(even) {{ background: #ffe8a0; }}
    td.path {{ font-family: monospace; font-size: 0.8rem; word-break: break-all; }}
    td.ts {{ white-space: nowrap; font-size: 0.8rem; color: #555; }}
    td.dry {{ text-align: center; }}
    details summary {{ cursor: pointer; color: #333; }}
    a {{ color: #0066cc; }}
    a:hover {{ color: #004499; }}
  </style>
</head>
<body>
  <h1>sortAI Audit Log</h1>
  <p class="meta">
    Generated: {generated_at} &nbsp;|&nbsp;
    Total: <strong>{total}</strong> &nbsp;|&nbsp;
    Dry-run: <strong>{dry_run_count}</strong>
  </p>
  <input type="text" id="filterInput" placeholder="Filter rows…" oninput="filterRows()">
  <table id="logTable">
    <thead>
      <tr>
        <th onclick="sortTable(0)">Timestamp &#9651;</th>
        <th onclick="sortTable(1)">Original file</th>
        <th onclick="sortTable(2)">Destination file</th>
        <th onclick="sortTable(3)">Summary</th>
        <th onclick="sortTable(4)">Dry run</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <script>
    function filterRows() {{
      var f = document.getElementById('filterInput').value.toLowerCase();
      document.querySelectorAll('#logTable tbody tr').forEach(function(row) {{
        row.style.display = row.textContent.toLowerCase().includes(f) ? '' : 'none';
      }});
    }}
    var _sortDir = {{}};
    function sortTable(col) {{
      var tbody = document.querySelector('#logTable tbody');
      var rows = Array.from(tbody.rows);
      var dir = (_sortDir[col] === 'asc') ? 'desc' : 'asc';
      _sortDir[col] = dir;
      rows.sort(function(a, b) {{
        var at = a.cells[col].textContent.trim();
        var bt = b.cells[col].textContent.trim();
        return dir === 'asc' ? at.localeCompare(bt) : bt.localeCompare(at);
      }});
      rows.forEach(function(r) {{ tbody.appendChild(r); }});
    }}
  </script>
</body>
</html>"""


def _build_rows(entries: list[dict]) -> str:
    parts = []
    for e in entries:
        ts = _esc(e.get("timestamp", "")[:19])
        orig_path = e.get("original_path", "")
        new_path = e.get("new_path", "")
        summary = e.get("summary", "")
        is_dry = e.get("dry_run", False)

        orig_name = _esc(Path(orig_path).name) if orig_path else ""
        new_name = _esc(Path(new_path).name) if new_path else ""
        orig_path_attr = _esc(orig_path)
        new_path_attr = _esc(new_path)

        try:
            dest_uri = _esc(Path(new_path).as_uri())
        except ValueError:
            dest_uri = "#"

        short_summary = _esc(summary[:120] + ("…" if len(summary) > 120 else ""))
        full_summary = _esc(summary)
        dry_class = " dry-run" if is_dry else ""
        dry_text = "Yes" if is_dry else "No"

        parts.append(
            f'      <tr class="entry{dry_class}">\n'
            f'        <td class="ts">{ts}</td>\n'
            f'        <td class="path" title="{orig_path_attr}">{orig_name}</td>\n'
            f'        <td class="path" title="{new_path_attr}"><a href="{dest_uri}">{new_name}</a></td>\n'
            f'        <td><details><summary>{short_summary}</summary><p>{full_summary}</p></details></td>\n'
            f'        <td class="dry">{dry_text}</td>\n'
            f'      </tr>'
        )
    return "\n".join(parts)
