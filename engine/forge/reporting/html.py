"""Diagnostics as one self-contained HTML file.

Deliberately dependency-free and single-file: no template engine, no CDN, no
external stylesheet. A report that needs a network round trip to render is not
a report someone can attach to an email or open two years from now.

Everything user-derived is escaped. Vault content -- filenames, link targets,
heading text -- is data from an arbitrary folder on someone's disk, and a note
called ``<script>.md`` must not execute when its own health report is opened.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from typing import Any

from .summary import headline

_CSS = """
:root {
  --bg: #fbfaf9; --fg: #1c1a17; --muted: #6b6660; --line: #e3dfd9;
  --card: #ffffff; --ok: #1f7a4d; --bad: #b03030; --accent: #8a5a2b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16150f; --fg: #eceae4; --muted: #9a948c; --line: #302d26;
    --card: #1e1c16; --ok: #5fbf8f; --bad: #e8776a; --accent: #d9a05b;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  padding: 2.5rem 1.25rem 4rem;
}
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
h2 { font-size: 1.05rem; margin: 2.5rem 0 .75rem; letter-spacing: -.005em; }
.sub { color: var(--muted); margin: 0 0 2rem; }
.cards { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); }
.card {
  background: var(--card); border: 1px solid var(--line);
  border-radius: .6rem; padding: .9rem 1rem;
}
.card .label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
.card .value { font-size: 1.35rem; font-weight: 600; margin-top: .3rem; }
.card.ok .value { color: var(--ok); }
.card.bad .value { color: var(--bad); }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
code {
  font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  background: color-mix(in srgb, var(--fg) 7%, transparent);
  padding: .1rem .3rem; border-radius: .25rem;
}
.wrap { overflow-x: auto; }
.verdict { font-size: 1.05rem; margin: 1.5rem 0 0; padding: .9rem 1rem;
  border-left: 3px solid var(--accent); background: var(--card); border-radius: 0 .4rem .4rem 0; }
.muted { color: var(--muted); }
footer { margin-top: 3.5rem; padding-top: 1.25rem; border-top: 1px solid var(--line);
  color: var(--muted); font-size: .85rem; }
a { color: var(--accent); }
"""


def _table(headers: list[str], rows: list[list[str]], *, numeric: set[int] = frozenset()) -> str:
    if not rows:
        return ""
    head = "".join(
        f'<th class="num">{escape(h)}</th>' if i in numeric else f"<th>{escape(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="num">{cell}</td>' if i in numeric else f"<td>{cell}</td>"
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_html(payload: dict[str, Any], *, vault_name: str = "vault") -> str:
    facts = headline(payload)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cards = "".join(
        f'<div class="card {"ok" if ok else "bad"}">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div></div>'
        for label, value, ok in facts.rows()
    )

    verdict = (
        "Nothing broken found. Every wikilink resolves and every convention "
        "the vault declares is satisfied."
        if facts.clean
        else "This vault has problems its author cannot see by reading it."
    )

    sections: list[str] = []

    unresolved = payload.get("links", {}).get("unresolved_targets", {})
    if unresolved:
        rows = []
        for target, info in sorted(unresolved.items(), key=lambda kv: -kv[1]["count"])[:100]:
            sources = info.get("sources", [])
            shown = ", ".join(f"<code>{escape(s)}</code>" for s in sources[:3])
            if len(sources) > 3:
                shown += f' <span class="muted">+{len(sources) - 3} more</span>'
            rows.append([f"<code>{escape(target)}</code>", str(info["count"]), shown])
        sections.append(
            "<h2>Dead links</h2>"
            '<p class="muted">Targets that are linked to but do not exist.</p>'
            + _table(["Target", "Times", "Referenced from"], rows, numeric={1})
        )

    fm = payload.get("frontmatter", {})
    by_code = fm.get("by_code", {})
    if by_code:
        descriptions = fm.get("code_descriptions", {})
        rows = [
            [escape(code), str(count), escape(descriptions.get(code, ""))]
            for code, count in sorted(by_code.items(), key=lambda kv: -kv[1])
        ]
        sections.append(
            "<h2>Frontmatter</h2>" + _table(["Code", "Files", "Meaning"], rows, numeric={1})
        )

    coverage = fm.get("coverage_by_folder", {})
    if coverage:
        rows = []
        for folder, info in sorted(
            coverage.items(), key=lambda kv: kv[1]["with_fm"] / max(kv[1]["total"], 1)
        )[:20]:
            total, have = info["total"], info["with_fm"]
            rows.append(
                [escape(folder), str(total), str(have), f"{100 * have / max(total, 1):.0f}%"]
            )
        sections.append(
            "<h2>Frontmatter coverage by folder</h2>"
            '<p class="muted">Least covered first.</p>'
            + _table(["Folder", "Files", "With frontmatter", "Coverage"], rows, numeric={1, 2, 3})
        )

    conventions = payload.get("conventions", {})
    if conventions:
        status = escape(conventions.get("resolution_status", ""))
        body = f'<p class="muted">{status}</p>'
        conflicts = conventions.get("conflicts", [])
        if conflicts:
            rows = [
                [
                    escape(c["kind"]),
                    f"<code>{escape(c['repo_wide'])}</code>",
                    f"<code>{escape(c['dsa_local'])}</code>",
                ]
                for c in conflicts
            ]
            body += _table(["Field", "One system says", "The other says"], rows)
        sections.append("<h2>Conventions</h2>" + body)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vault health — {escape(vault_name)}</title>
<style>{_CSS}</style></head>
<body><main>
<h1>Vault health — {escape(vault_name)}</h1>
<p class="sub">{facts.files:,} Markdown files · {facts.links:,} links · {escape(generated)}</p>
<div class="cards">{cards}</div>
<p class="verdict">{escape(verdict)}</p>
{"".join(sections)}
<footer>
Generated by <a href="https://pypi.org/project/forge-kb/">forge-kb</a> —
<code>pip install forge-kb</code>, then <code>forge index &amp;&amp; forge diagnostics</code>.
No model, no API key, no network. The vault is never modified.
</footer>
</main></body></html>
"""
