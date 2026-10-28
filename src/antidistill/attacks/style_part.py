#!/usr/bin/env python
"""Style attack on PART -- rewrite PART's solutions with invented checks and corrections.

Design, checks and the pre-registered rule: results/m6f_style_part_design.md.

Why: the re-voice attack (results/REPORT.md Part 9) restored PART's first-person voice but not its
uncertainty, and recovered nothing. PART's step 1 does more than change the voice: it collapses each
checking episode into its conclusion, so the doubt and the re-checking are gone from the text. This
attack therefore asks the rewriter to put checking back -- to consider an approach before committing,
note where a step could go wrong, double back and verify -- which is what our Style attack did against
hindsight (64.2%, the 32B attacker), applied piece by piece to PART's full-length solutions.

Same threat model and engine as the re-voice attack (antidistill.attacks.revoice): PART's 800 defended
solutions, an ordinary instruction model, ~1,024-token pieces, the final answer copied unchanged. The
checks differ in one way: added checks make text longer, so the upper length bound is wider.

Usage:
    python -m antidistill.attacks.style_part --input data/defended/limo_part.json \
        --output data/curated/style_part.json --base-url http://127.0.0.1:8003/v1 [--pilot 20]
"""
from __future__ import annotations

from antidistill.attacks.revoice import Revoice, main as _main

# The Style prompt (results/m6b_prompts.md, A1), adapted from whole solutions to pieces of one:
# "a polished SOLUTION" -> "a part of a complete reasoning process", and the answer instruction dropped
# because the final-answer section is never sent.
STYLE_PART_PROMPT_V1 = (
    "Rewrite the given text, which is a part of a complete reasoning process, as a first-person "
    "account of working the problem out for the first time. Show the reasoning as it would actually "
    "unfold: consider an approach before committing to it, note where a step could go wrong, double "
    "back when something does not check out, and verify results that are easy to get wrong, for "
    "example \"Wait, ...\", \"Hmm, ...\", \"Let me check that...\", \"Maybe ...\", \"Actually, ...\". "
    "Keep every calculation, number and result from the original text.\n"
    "Do not include any introductory phrases.\n"
    "Text:"
)
# Revision 1 of at most 2 (m6f §5), 2026-10-12, after the first pilot and before any training: with V1 the
# instruction model CONDENSED the text (rejected attempts came back at ~0.72x the input; 8.6% of pieces
# had every attempt rejected) and added almost no uncertainty words (11.9 -> 13.1 per 1k; LIMO 36.5).
# This version says to keep everything and names where to add doubt.
STYLE_PART_PROMPT_V2 = (
    "Rewrite the given text, which is a part of a complete reasoning process, as a first-person "
    "account of working the problem out for the first time. Keep all of the original content: every "
    "sentence's content, every calculation, every number and every result, so the rewrite is longer "
    "than the original. Add what a person working it out would say along the way, several times in "
    "the text: before a step, briefly wonder whether the approach is right (\"Hmm, maybe I should "
    "...\", \"Let me think...\"); after a result that is easy to get wrong, check it (\"Wait, let "
    "me check that...\", \"Actually, ...\"); where the text settles a question, show the doubt that "
    "came first (\"Wait, is that right?\"). Do not repeat the same check twice.\n"
    "Do not include any introductory phrases.\n"
    "Text:"
)
# Revision 2 of at most 2 (m6f §5), 2026-10-12, after the second pilot and before any training: V2 fixed
# the condensing (1.2% of pieces rejected) but reached 17.2 uncertainty words per 1k, 47% of LIMO's where
# 50% is required. Word by word the shortfall was "wait" (3.4 vs LIMO 10.9), "perhaps" (1.7 vs 4.8) and
# "maybe" (3.5 vs 6.4); the rewrite also dropped part of PART's own "alternatively" (5.5 -> 3.8), and only
# 57% of paragraphs received any added doubt. This version asks for doubt in every paragraph and for the
# original's "alternatively"/"perhaps" passages to be kept. No further revision is allowed.
STYLE_PART_PROMPT = (
    "Rewrite the given text, which is a part of a complete reasoning process, as a first-person "
    "account of working the problem out for the first time. Keep all of the original content: every "
    "sentence's content, every calculation, every number and every result, so the rewrite is longer "
    "than the original. Where the original considers another method or another possibility "
    "(\"Alternatively, ...\", \"Perhaps ...\"), keep that wording. In every paragraph, add what a "
    "person working it out would say along the way: before a step, wonder whether the approach is "
    "right (\"Hmm, maybe I should ...\", \"Perhaps ...\"); after a result, stop and question it "
    "(\"Wait, is that right? Let me check...\", \"Wait, actually ...\"); where the text settles a "
    "question, show the doubt that came first (\"Wait, maybe ...\"). Do not repeat the same check "
    "twice.\n"
    "Do not include any introductory phrases.\n"
    "Text:"
)


class StylePart(Revoice):
    name = "style_part"
    prompt = STYLE_PART_PROMPT
    ratio = (0.90, 2.20)       # added checks lengthen the text; summarising is still rejected
    content_min = 0.70


if __name__ == "__main__":
    raise SystemExit(_main(StylePart, "data/curated/style_part.json", __doc__))
