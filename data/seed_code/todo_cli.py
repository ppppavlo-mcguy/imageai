# instruction: write a python todo command line app
import json
import sys
from pathlib import Path

STORE = Path("todos.json")


def load() -> list:
    return json.loads(STORE.read_text()) if STORE.exists() else []


def save(items: list) -> None:
    STORE.write_text(json.dumps(items, indent=2))


def main() -> None:
    args = sys.argv[1:]
    items = load()
    if not args or args[0] == "list":
        for i, item in enumerate(items, 1):
            print(f"{i}. {item}")
    elif args[0] == "add":
        items.append(" ".join(args[1:]))
        save(items)
        print("added")
