#!/usr/bin/env python3
"""Regenerate the dashboard screenshots in docs/media/.

    python scripts/screenshot.py [vault]

With no argument it writes a demo vault to ~/notebook and shoots that, so the
images in the README show exactly what `forge demo` produces on the reader's
own machine. The path is fixed rather than temporary because the dashboard
prints it on screen, and a temporary directory would put a different random
string in the committed image every time it is regenerated.

Pass a path to shoot a real vault instead; look at what is on screen before
committing that, because a screenshot of a private vault publishes its page
titles.

Textual renders the SVG itself, through the same `save_screenshot` the library
offers, so these are the real terminal output rather than a mockup. The PNGs
beside them are those SVGs rendered in a browser: GitHub's Markdown sanitiser
does not reliably display an SVG this complex, and a README whose only image
is a broken one is worse than a README with no image.

Needs the `tui` extra: pip install -e ".[dev,tui]"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from forge.cli.dashboard import build_dashboard  # noqa: E402
from forge.cli.demo import write_vault  # noqa: E402
from forge.cli.snapshot import build_snapshot  # noqa: E402
from forge.config import Settings  # noqa: E402

MEDIA = Path(__file__).resolve().parents[1] / "docs" / "media"

#: name -> (columns, rows, key to press first). The sizes are chosen so each
#: screen fills its frame: the overview is tall, the issues table is wide
#: enough that the Hint column is not cut off mid-word.
SHOTS: dict[str, tuple[int, int, str | None]] = {
    "dash-overview": (104, 34, None),
    "dash-issues": (126, 14, "4"),
}


async def shoot(settings: Settings, name: str, cols: int, rows: int, key: str | None) -> Path:
    app = build_dashboard(settings, build_snapshot(settings))
    out = MEDIA / f"{name}.svg"
    async with app.run_test(size=(cols, rows)) as pilot:
        if key:
            await pilot.press(key)
        await pilot.pause()
        app.save_screenshot(str(out))
    return out


def main() -> int:
    if len(sys.argv) > 1:
        vault = Path(sys.argv[1]).resolve()
    else:
        vault = Path.home() / "notebook"
        vault.mkdir(parents=True, exist_ok=True)
        write_vault(vault)
        print(f"demo vault: {vault}")

    MEDIA.mkdir(parents=True, exist_ok=True)
    settings = Settings(vault_path=vault, state_dir=vault / ".forge")

    # Indexing and seeding first, or the dashboard opens on an empty store and
    # every panel reads zero.
    from forge.bootstrap import build_plan
    from forge.corpus import IndexPipeline, load_store
    from forge.corpus.indexer import CorpusIndexer

    store = load_store(settings)
    IndexPipeline(settings, store).run(persist=True, write_reports=False)
    plan = build_plan(CorpusIndexer(settings).build_index())
    for concept in plan.concepts:
        store.put_concept(concept)
    for link in plan.links:
        store.put_link(link)
    store.set_concept_inbound(plan.inbound_links)
    store.close()

    for name, (cols, rows, key) in SHOTS.items():
        print(f"wrote {asyncio.run(shoot(settings, name, cols, rows, key))}")
    print("\nPNGs are these SVGs rendered in a browser; regenerate them the same way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
