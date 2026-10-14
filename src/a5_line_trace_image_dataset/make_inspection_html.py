"""Build a side-by-side HTML index for a PNG-mode line-tracing split."""

from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--output_html", type=Path, required=True)
    parser.add_argument("--title", default="A5 line-tracing eval inspection")
    return parser.parse_args()


def image_url(image_ref: str, data_root: Path, html_dir: Path) -> str:
    if image_ref.startswith("shard:"):
        raise SystemExit("Inspection HTML requires PNG-mode data, not shard references")
    image_path = (data_root / image_ref).resolve()
    relative_path = os.path.relpath(image_path, html_dir.resolve()).replace(os.sep, "/")
    return quote(relative_path, safe="/")


def main() -> None:
    args = parse_args()
    metadata_csv = args.metadata_csv.resolve()
    output_html = args.output_html.resolve()
    data_root = metadata_csv.parent

    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows found in {metadata_csv}")

    output_html.parent.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for position, row in enumerate(rows):
        input_url = image_url(row["edit_image"], data_root, output_html.parent)
        target_url = image_url(row["image"], data_root, output_html.parent)
        sample_index = html.escape(row.get("index", str(position)))
        length = html.escape(row.get("num_permutations", "?"))
        width = html.escape(row.get("image_width", "?"))
        cards.append(
            f"""
            <section class="sample">
              <h2>#{sample_index} <span>length={length}, width={width}px</span></h2>
              <div class="pair">
                <figure><figcaption>Input</figcaption><a href="{input_url}"><img src="{input_url}" loading="lazy"></a></figure>
                <figure><figcaption>Target</figcaption><a href="{target_url}"><img src="{target_url}" loading="lazy"></a></figure>
              </div>
            </section>
            """
        )

    title = html.escape(args.title)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin: 24px; color: #202124; background: #f5f6f8; font: 14px/1.4 system-ui, sans-serif; }}
    header {{ margin: 0 auto 20px; max-width: 1600px; }}
    .sample {{ margin: 0 auto 18px; padding: 14px; max-width: 1600px; background: white; border: 1px solid #d9dce1; border-radius: 8px; }}
    h1 {{ margin: 0 0 4px; }}
    h2 {{ margin: 0 0 10px; font-size: 15px; }}
    h2 span {{ margin-left: 8px; color: #5f6368; font-weight: 400; }}
    .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    figure {{ margin: 0; min-width: 0; }}
    figcaption {{ margin-bottom: 5px; font-weight: 600; }}
    img {{ display: block; width: 100%; height: auto; image-rendering: auto; border: 1px solid #eceff1; }}
    @media (max-width: 900px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header><h1>{title}</h1><div>{len(rows)} input/target pairs. Click an image to view it at full resolution.</div></header>
  {''.join(cards)}
</body>
</html>
"""
    output_html.write_text(document, encoding="utf-8")
    print(f"Inspection page: {output_html} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
