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
    archive_root: Path | None = None,
) -> None:
    """Append a JSON-lines entry to *log_path* and regenerate the HTML report."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "original_path": str(src.resolve()),
        "new_path": str(dest.resolve()),
        "archive_root": str(archive_root.resolve()) if archive_root else None,
        "summary": summary,
        "dry_run": dry_run,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    render_html_report(log_path)


def load_jsonl_entries(log_path: Path) -> list[dict]:
    """Parse all valid JSON-lines entries from *log_path*, skipping malformed lines."""
    entries: list[dict] = []
    if not log_path.exists():
        return entries
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line := line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def render_html_report(log_path: Path) -> None:
    """Read all JSONL entries from *log_path* and write a self-contained HTML report."""
    entries = load_jsonl_entries(log_path)
    html_path = _html_path(log_path)
    html_path.write_text(_build_html(entries), encoding="utf-8")


def dest_label(new_path: str, archive_root_str: str | None) -> str:
    """Return the display label for a destination path.

    Shows the path relative to the archive root when available, otherwise
    just the filename.
    """
    if not new_path:
        return ""
    p = Path(new_path)
    if archive_root_str:
        try:
            return Path(new_path).relative_to(archive_root_str).as_posix()
        except ValueError:
            pass
    return p.name


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _html_path(log_path: Path) -> Path:
    return log_path.with_name(log_path.stem + "_report.html")


def _build_html(entries: list[dict]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(entries)
    dry_run_count = sum(1 for e in entries if e.get("dry_run"))
    rows = _build_rows(reversed(entries))

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
        <th onclick="sortTable(0)">Timestamp</th>
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
    var _sortDir = {{0: 'desc'}};
    var _labels = ['Timestamp','Original file','Destination file','Summary','Dry run'];
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
      document.querySelectorAll('#logTable thead th').forEach(function(th, i) {{
        th.textContent = _labels[i] + (i === col ? (dir === 'asc' ? ' △' : ' ▽') : '');
      }});
    }}
    document.addEventListener('DOMContentLoaded', function() {{
      document.querySelector('#logTable thead th').textContent = _labels[0] + ' ▽';
    }});
  </script>
</body>
</html>"""


def _build_rows(entries) -> str:
    parts = []
    for e in entries:
        ts = _esc(e.get("timestamp", "")[:19])
        orig_path = e.get("original_path", "")
        new_path = e.get("new_path", "")
        summary = e.get("summary", "")
        is_dry = e.get("dry_run", False)

        orig_name = _esc(Path(orig_path).name) if orig_path else ""
        orig_path_attr = _esc(orig_path)
        new_path_attr = _esc(new_path)
        label = _esc(dest_label(new_path, e.get("archive_root")))

        new_p = Path(new_path) if new_path else Path()
        try:
            dest_uri = _esc(new_p.as_uri())
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
            f'        <td class="path" title="{new_path_attr}"><a href="{dest_uri}">{label}</a></td>\n'
            f'        <td><details><summary>{short_summary}</summary><p>{full_summary}</p></details></td>\n'
            f'        <td class="dry">{dry_text}</td>\n'
            f'      </tr>'
        )
    return "\n".join(parts)
