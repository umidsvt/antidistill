"""Second task (results/m8_audit_design.md §34): answer parsing and scoring for the Reasoning Gym tasks, in plain Python
so that both .venv-rg (puzzle building) and .venv-dg (vLLM) can import it.

Rows (from scripts/rg_build.py): {"id", "task": "zebra" | "crypt", "level", "question", "answer", ...}
  zebra: "categories" (labels, e.g. Name, Animal), "values" (per category, as listed in the puzzle), "solution"
         (per house, one value per category)
  crypt: "letters" (sorted), "solution" ({letter: digit})
"""
from __future__ import annotations

import re

ZEBRA_FORMAT = ("Determine the full arrangement. At the end of your answer, give one line per house, in this format, "
                "with the values spelled exactly as listed above:\n{example}")


def final_part(text: str, reasoning: bool) -> str | None:
    """The answer part of a reply: after a reasoning model's last </think> (None if it never finished thinking)."""
    if reasoning:
        return text.rsplit("</think>", 1)[1] if "</think>" in text else None
    return text.rsplit("</think>", 1)[1] if "</think>" in text else text


def _norm(s: str) -> str:
    return re.sub(r"[\s*_`\"'.]+", " ", s.lower()).strip()


def parse_zebra(text: str, row: dict, reasoning: bool = False) -> list[list[str]] | None:
    """The grid from the last 'House i: Cat=value, ...' line of each house; None if any house or cell is missing."""
    part = final_part(text, reasoning)
    if part is None:
        return None
    cats = [c.lower() for c in row["categories"]]
    vals = [{_norm(v): v for v in vs} for vs in row["values"]]
    k = len(row["solution"])
    grid: dict[int, list[str | None]] = {}
    for m in re.finditer(r"House\s*#?\s*(\d+)\s*[:\-]\s*(.+)", part):
        h = int(m.group(1))
        if not 1 <= h <= k:
            continue
        cells: list[str | None] = [None] * len(cats)
        for pair in re.split(r"[,;|]", m.group(2)):
            if "=" not in pair and ":" not in pair:
                continue
            key, _, val = re.split(r"([=:])", pair, maxsplit=1)
            key, val = _norm(key), _norm(val)
            for j, c in enumerate(cats):
                if key == c or key.endswith(" " + c) or c in key.split():
                    cells[j] = vals[j].get(val)
        grid[h] = cells   # a later line for the same house replaces an earlier one
    if len(grid) != k or any(c is None for h in grid for c in grid[h]):
        return None
    return [grid[h] for h in range(1, k + 1)]


def parse_crypt(text: str, row: dict, reasoning: bool = False) -> dict[str, int] | None:
    """The last complete letter=digit mapping in the answer part; None if no line maps every letter."""
    part = final_part(text, reasoning)
    if part is None:
        return None
    letters = set(row["letters"])
    best = None
    for line in part.splitlines() + [part.replace("\n", ",")]:
        pairs = re.findall(r"\b([A-Z])\s*=\s*(\d)\b", line)
        if pairs and {p[0] for p in pairs} >= letters:
            best = {p: int(d) for p, d in pairs if p in letters}
    return best


def parse(text: str, row: dict, reasoning: bool = False):
    return (parse_zebra if row["task"] == "zebra" else parse_crypt)(text, row, reasoning)


def correct(parsed, row: dict) -> bool:
    if parsed is None:
        return False
    if row["task"] == "zebra":
        return [[_norm(x) for x in h] for h in parsed] == [[_norm(x) for x in h] for h in row["solution"]]
    return parsed == {k: int(v) for k, v in row["solution"].items()}


def cell_accuracy(parsed, row: dict) -> float:
    """Share of cells (zebra) or letters (crypt) right; 0 for an unreadable answer."""
    if parsed is None:
        return 0.0
    if row["task"] == "zebra":
        pairs = [(_norm(a), _norm(b)) for ph, sh in zip(parsed, row["solution"]) for a, b in zip(ph, sh)]
    else:
        pairs = [(parsed.get(k), int(v)) for k, v in row["solution"].items()]
    return sum(a == b for a, b in pairs) / len(pairs)
