#!/usr/bin/env python
"""Post-hoc stripping of epistemic verbalization — the density/length decoupling condition.

Design, and the two approaches already ruled out: results/decoupling_design.md.

Epistemic density and trace length are collinear across our pools (r = 0.978) and within each
problem (r = +0.826), and prompting cannot separate them: asked for long-but-confident output, R1
either keeps searching (and doubting) or stops searching (and gets short). So instead of asking a
model, we edit text: delete the epistemic content from the longest, densest pool in place.

Three levels, increasingly aggressive, so the length/density frontier is measured not assumed:

  S1 marker   the epistemic tokens themselves, plus adjoining connective punctuation
  S2 clause   the clause containing a marker, bounded by , ; : or sentence end
  S3 sentence the whole sentence containing a marker

Usage:
    python -m antidistill.attacks.strip_epistemic --level s1 \
        --input data/curated/recon7b_solo.json --output data/curated/strip_s1.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# The nine epistemic markers used throughout the project's density metric. Kept identical so a
# stripped trace's measured density is directly comparable with every other pool.
MARKERS = ["let me reconsider", "on second thought", "alternatively", "perhaps", "actually",
           "maybe", "i think", "wait", "hmm"]
_MARK = re.compile(r"\b(" + "|".join(re.escape(m) for m in MARKERS) + r")\b", re.I)

# HELD-OUT reconsideration lexicon. The stripper NEVER targets these, which is the whole point:
# the nine MARKERS are exactly what S1 deletes, so measuring density with them after stripping is
# circular (it drops to ~0 by construction). This list is the independent instrument.
#
# FROZEN 2026-09-17. Do not edit it after seeing results -- extending it post hoc to catch "what
# remains" is exactly the whack-a-mole that makes the measurement meaningless. If it must change,
# add a new versioned list and report both.
HELD_OUT = [r"let me (check|verify|double[- ]check|re-?examine|recheck|re-?calculate|try)",
            r"(that|this|it) (doesn'?t|does not|can'?t|cannot) (work|be right|be correct)",
            r"(i made|made) a mistake", r"\bhold on\b", r"\bno,\s", r"\blet'?s try\b",
            r"\binstead\b", r"\bis that (right|correct)\b", r"\bdid i\b",
            r"\bnot (quite|right|correct)\b", r"\bon (the )?other hand\b", r"\bor maybe\b",
            r"\bi('m| am) (not sure|confused|stuck)\b", r"\bdouble[- ]check", r"\bthat can'?t be\b"]
_HELD = re.compile("|".join(HELD_OUT), re.I)


def marker_density(text: str) -> float:
    """The nine-marker density used everywhere else. CIRCULAR on stripped text -- see HELD_OUT."""
    return len(_MARK.findall(text or "")) / max(len((text or "").split()), 1) * 1000


def held_out_density(text: str) -> float:
    """Reconsideration language the stripper never removes. The honest post-strip measure."""
    return len(_HELD.findall(text or "")) / max(len((text or "").split()), 1) * 1000


# Mathematics must never be touched. Protect LaTeX spans so a clause/sentence split cannot cut
# through `\( ... \)`, `\[ ... \]`, `$...$` or a `\boxed{...}`.
_MATH = re.compile(r"(\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$[^$\n]*?\$|\\boxed\{(?:[^{}]|\{[^{}]*\})*\})",
                   re.S)


def _protect(text: str):
    spans = []
    def keep(m):
        spans.append(m.group(0))
        return f"\x00{len(spans)-1}\x00"
    return _MATH.sub(keep, text), spans


def _restore(text: str, spans) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)


def strip_s1(text: str) -> str:
    """Remove the marker words only, plus a comma/colon that becomes dangling."""
    t, spans = _protect(text)
    t = re.sub(r"(?i)\b(" + "|".join(re.escape(m) for m in MARKERS) + r")\b[\s,:;.!?-]*", "", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"(^|\n)\s*[,;:]\s*", r"\1", t)          # line now starting with punctuation
    t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return _restore(t, spans)


def strip_s2(text: str) -> str:
    """Remove each clause that contains a marker (clause = run between , ; : or sentence end)."""
    t, spans = _protect(text)
    out = []
    for sent in re.split(r"(?<=[.!?])(\s+)", t):
        if not _MARK.search(sent):
            out.append(sent); continue
        parts = re.split(r"([,;:])", sent)
        kept, i = [], 0
        while i < len(parts):
            clause = parts[i]; sep = parts[i + 1] if i + 1 < len(parts) else ""
            if not _MARK.search(clause):
                kept.append(clause + sep)
            i += 2
        s = "".join(kept).strip(" ,;:")
        if s and not re.search(r"[.!?]$", s):
            s += "."
        out.append(s)
    t = "".join(out)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return _restore(t, spans)


def strip_s3(text: str) -> str:
    """Remove every sentence that contains a marker."""
    t, spans = _protect(text)
    out = [s for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip() and not _MARK.search(s)]
    return _restore("\n".join(out), spans)


LEVELS = {"s1": strip_s1, "s2": strip_s2, "s3": strip_s3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=list(LEVELS), required=True)
    ap.add_argument("--input", default="data/curated/recon7b_solo.json")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    rows = json.loads(Path(a.input).read_text(encoding="utf-8"))
    f = LEVELS[a.level]
    out = []
    for r in rows:
        n = dict(r); n["output"] = f(r["output"] or "")
        out.append(n)
    Path(a.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{a.level}: {len(out)} traces -> {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
