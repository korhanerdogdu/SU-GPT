from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.config import SOURCES_DIR
from modules.source_indexer import ingest_source_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Index sources/ course modules into Chroma.")
    parser.add_argument("--sources-dir", default=SOURCES_DIR)
    args = parser.parse_args()
    count = ingest_source_directory(args.sources_dir)
    print(f"Indexed source chunks: {count}")


if __name__ == "__main__":
    main()
