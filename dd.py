#!/usr/bin/env python3
"""Download every file from an Apache-style directory index into a local folder.

Usage:
    python download_dir.py [URL] [OUTPUT_DIR]

Defaults to the PROBA2/SWAP IDL prep directory and ./swap_prep.
Standard library only.
"""
import os
import sys
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, unquote

# DEFAULT_URL = "https://hesperia.gsfc.nasa.gov/ssw/proba2/swap/idl/prep/"
# DEFAULT_URL = "https://hesperia.gsfc.nasa.gov/ssw/proba2/swap/caldb/"
# DEFAULT_URL = "https://hesperia.gsfc.nasa.gov/ssw/proba2/swap/idl/tools/"
DEFAULT_URL = "https://hesperia.gsfc.nasa.gov/ssw/proba2/swap/idl/objects/"
# DEFAULT_OUT = "swap_prep"
# DEFAULT_OUT = "swap_caldb"
# DEFAULT_OUT = "swap_tools"
DEFAULT_OUT = "swap_objects"


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.links.append(value)


def list_files(index_url):
    """Return absolute URLs of the files (not subdirs/sort/parent links) in the index."""
    with urllib.request.urlopen(index_url) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(html)

    base_path = urlparse(index_url).path
    files = []
    for href in parser.links:
        if href.startswith(("?", "#")) or href.endswith("/"):
            continue  # column-sort links, anchors, directories, parent dir
        full = urljoin(index_url, href)
        parsed = urlparse(full)
        # Keep only files that live directly in this directory
        if parsed.netloc != urlparse(index_url).netloc:
            continue
        if os.path.dirname(parsed.path) + "/" != base_path:
            continue
        files.append(full)
    return sorted(set(files))


def download(url, out_dir):
    name = unquote(os.path.basename(urlparse(url).path))
    dest = os.path.join(out_dir, name)
    urllib.request.urlretrieve(url, dest)
    return dest


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    if not url.endswith("/"):
        url += "/"

    os.makedirs(out_dir, exist_ok=True)
    files = list_files(url)
    print(f"Found {len(files)} files at {url}")

    for i, f in enumerate(files, 1):
        try:
            dest = download(f, out_dir)
            print(f"[{i}/{len(files)}] {dest}")
        except Exception as exc:
            print(f"[{i}/{len(files)}] FAILED {f}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()