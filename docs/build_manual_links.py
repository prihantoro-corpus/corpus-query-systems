"""Generate stable GitHub Pages redirect URLs for the CORTEX manual.

The manual content lives in Google Docs (edited live). This script creates
permanent addresses such as
    https://prihantoro-corpus.github.io/<repo>/manual/concordance/
that redirect to the corresponding Google Doc, so URLs cited in papers never change.

Usage:  python docs/build_manual_links.py
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).parent
MANUAL = ROOT / "manual"

REDIRECT = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title} | CORTEX Manual</title>
<meta http-equiv="refresh" content="0; url={url}">
<link rel="canonical" href="{url}">
<script>location.replace({url_js});</script>
</head><body>
<p>Redirecting to <a href="{url}">{title}</a>&hellip;</p>
</body></html>
"""

INDEX = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CORTEX Manual</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:720px;margin:3rem auto;padding:0 1rem;line-height:1.6;color:#222}}
h1{{margin-bottom:.2rem}} .sub{{color:#666;margin-top:0}}
li{{margin:.4rem 0}} a{{color:#2b59c3;text-decoration:none}} a:hover{{text-decoration:underline}}
footer{{margin-top:3rem;font-size:.85rem;color:#666}}
</style></head><body>
<h1>CORTEX Manual</h1>
<p class="sub">User documentation for CORTEX: Advanced Corpus Query System</p>
<ol start="0">
{items}
</ol>
<footer>
Documentation &copy; Prihantoro, licensed under
<a href="https://creativecommons.org/licenses/by-nc/4.0/">CC BY-NC 4.0</a>.
Source code: <a href="https://github.com/prihantoro-corpus/cortex">GitHub</a> (AGPL-3.0).
</footer>
</body></html>
"""


def main() -> None:
    cfg = json.loads((MANUAL / "links.json").read_text(encoding="utf-8"))
    fallback = cfg["fallback"]
    items = []
    for ch in cfg["chapters"]:
        url = ch["url"] or fallback
        title = ch["title"]
        out = MANUAL / ch["slug"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "index.html").write_text(
            REDIRECT.format(title=html.escape(title), url=html.escape(url),
                            url_js=json.dumps(url)),
            encoding="utf-8")
        items.append(f'<li><a href="{ch["slug"]}/">{html.escape(title)}</a></li>')
        print(f"manual/{ch['slug']}/ -> {url}")
    (MANUAL / "index.html").write_text(INDEX.format(items="\n".join(items)),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
