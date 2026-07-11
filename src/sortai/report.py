"""HTML audit-report generation from the JSONL decision log."""

from __future__ import annotations

from datetime import datetime
from html import escape as _esc
from pathlib import Path

from sortai.file_ops import load_jsonl_entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    error_count = sum(1 for e in entries if e.get("error"))
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
    th.no-sort {{ cursor: default; }}
    th.no-sort:hover {{ background: #2c3e50; }}
    td {{ padding: 0.4rem 0.6rem; vertical-align: top; font-size: 0.85rem; border-bottom: 1px solid #e0e0e0; }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    tr.dry-run {{ background: #fff3cd; }}
    tr.dry-run:nth-child(even) {{ background: #ffe8a0; }}
    tr.error {{ background: #fde8e8; }}
    tr.error:nth-child(even) {{ background: #f8d0d0; }}
    .error-label {{ color: #c0392b; font-style: italic; }}
    td.path {{ font-family: monospace; font-size: 0.8rem; word-break: break-all; }}
    td.ts {{ white-space: nowrap; font-size: 0.8rem; color: #555; }}
    td.dry {{ text-align: center; }}
    details summary {{ cursor: pointer; color: #333; }}
    a {{ color: #0066cc; }}
    a:hover {{ color: #004499; }}
    .interactions {{ margin-top: 0.4rem; }}
    .interaction {{ margin-bottom: 0.6rem; border-left: 3px solid #ccc; padding-left: 0.5rem; }}
    .interaction h4 {{ margin: 0 0 0.2rem; font-size: 0.8rem; color: #2c3e50; }}
    .interaction details {{ margin-top: 0.2rem; }}
    .interaction pre {{ margin: 0.2rem 0 0; white-space: pre-wrap; font-size: 0.75rem; background: #f4f4f4; padding: 0.3rem 0.5rem; border-radius: 3px; max-height: 300px; overflow-y: auto; }}
  </style>
</head>
<body>
  <h1>sortAI Audit Log</h1>
  <p class="meta">
    Generated: {generated_at} &nbsp;|&nbsp;
    Total: <strong>{total}</strong> &nbsp;|&nbsp;
    Dry-run: <strong>{dry_run_count}</strong> &nbsp;|&nbsp;
    Errors: <strong>{error_count}</strong>
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
        <th class="no-sort">LLM interactions</th>
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
        is_dry = e.get("dry_run", False)
        is_error = e.get("error", False)

        orig_name = _esc(Path(orig_path).name) if orig_path else ""
        orig_path_attr = _esc(orig_path)

        if is_error:
            error_reason = _esc(e.get("error_reason", "unknown"))
            dest_cell = '<span class="error-label">[classification error]</span>'
            short_summary = f'<span class="error-label">{error_reason[:120]}</span>'
            full_summary = error_reason
            summary_cell = f'<details><summary>{short_summary}</summary><p>{_esc(e.get("error_reason", ""))}</p></details>'
        else:
            summary = e.get("summary", "")
            new_path_attr = _esc(new_path)
            label = _esc(dest_label(new_path, e.get("archive_root")))
            new_p = Path(new_path) if new_path else Path()
            try:
                dest_uri = _esc(new_p.as_uri())
            except ValueError:
                dest_uri = "#"
            dest_cell = f'<td class="path" title="{new_path_attr}"><a href="{dest_uri}">{label}</a></td>'
            short_summary = _esc(summary[:120] + ("…" if len(summary) > 120 else ""))
            full_summary = _esc(summary)
            summary_cell = f'<details><summary>{short_summary}</summary><p>{full_summary}</p></details>'

        interactions_cell = _build_interactions_cell(e.get("interactions", []))

        dry_class = " dry-run" if is_dry else ""
        error_class = " error" if is_error else ""
        dry_text = "Yes" if is_dry else "No"

        if is_error:
            parts.append(
                f'      <tr class="entry{error_class}">\n'
                f'        <td class="ts">{ts}</td>\n'
                f'        <td class="path" title="{orig_path_attr}">{orig_name}</td>\n'
                f'        <td class="path">{dest_cell}</td>\n'
                f'        <td>{summary_cell}</td>\n'
                f'        <td class="dry">{dry_text}</td>\n'
                f'        <td>{interactions_cell}</td>\n'
                f'      </tr>'
            )
        else:
            parts.append(
                f'      <tr class="entry{dry_class}">\n'
                f'        <td class="ts">{ts}</td>\n'
                f'        <td class="path" title="{orig_path_attr}">{orig_name}</td>\n'
                f'        {dest_cell}\n'
                f'        <td>{summary_cell}</td>\n'
                f'        <td class="dry">{dry_text}</td>\n'
                f'        <td>{interactions_cell}</td>\n'
                f'      </tr>'
            )
    return "\n".join(parts)


def _build_interactions_cell(interactions: list) -> str:
    if not interactions:
        return '<span style="color:#aaa">—</span>'
    items = []
    for ix in interactions:
        stage = _esc(str(ix.get("stage", "")))
        step = _esc(str(ix.get("step", "")))
        prompt = _esc(str(ix.get("prompt", "")))
        answer = _esc(str(ix.get("answer", "")))
        reasoning = _esc(str(ix.get("reasoning", "")))
        reasoning_html = (
            f'<details><summary>Reasoning</summary><pre>{reasoning}</pre></details>'
            if reasoning else ""
        )
        items.append(
            f'<div class="interaction">'
            f'<h4>{stage} (step {step})</h4>'
            f'<details><summary>Prompt</summary><pre>{prompt}</pre></details>'
            f'{reasoning_html}'
            f'<details><summary>Answer</summary><pre>{answer}</pre></details>'
            f'</div>'
        )
    count = len(interactions)
    label = f"{count} interaction{'s' if count != 1 else ''}"
    return f'<details><summary>{label}</summary><div class="interactions">{"".join(items)}</div></details>'
