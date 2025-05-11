# file: extract_fast_dataset.py

import json

INPUT_FILE = "books_data/goodreads_books.json"
OUTPUT_FILE = "books_data/book_snippets.json"
MAX_LINES = 9000

book_snippets = {"title": [], "description": []}

with open(INPUT_FILE, "r", encoding="utf-8") as infile:
    for i, line in enumerate(infile):
        if i >= MAX_LINES:
            break
        try:
            book = json.loads(line)
            title = book.get("title", "").strip()
            desc = book.get("description", "").strip()
            if title and desc:
                book_snippets["title"].append(title)
                book_snippets["description"].append(desc)
        except json.JSONDecodeError:
            continue

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    json.dump(book_snippets, outfile, ensure_ascii=False, indent=2)

print(f"✅ Extracted {len(book_snippets['title'])} valid entries to {OUTPUT_FILE}")
