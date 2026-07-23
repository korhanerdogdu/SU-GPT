from __future__ import annotations

"""Force-seed the MongoDB course list from data/course_catalog/current.jsonl.

Usage: python server/scripts/seed_courses.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.mongodb import ensure_user, seed_courses_from_catalog, courses
from modules.config import ADMIN_USERNAME


async def main() -> None:
    await ensure_user(ADMIN_USERNAME)  # also verifies Mongo connectivity
    touched = await seed_courses_from_catalog(force=True)
    total = await courses.estimated_document_count()
    print(f"Seeded/updated {touched} course records. Collection now holds ~{total} courses.")


if __name__ == "__main__":
    asyncio.run(main())
