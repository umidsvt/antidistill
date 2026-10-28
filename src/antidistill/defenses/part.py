#!/usr/bin/env python
"""PART -- information-preserving antidistillation reformulation of reasoning traces.

Ding, Cui, Dong, Zheng & Wei (2025), "Information-Preserving Reformulation of Reasoning Traces for
Antidistillation", arXiv:2510.11545. (The proposal attributes it to "Chen et al."; that is wrong --
results/m4_part_design.md §0.) No code was released, so this is implemented from the paper's text.

Each trace is cut into segments, and every segment goes through two LLM passes using the paper's
Appendix B prompts verbatim:

  pass 1  remove self-talk -- convert first-person self-talk to declarative form, touch nothing else
  pass 2  reorder          -- hoist the most important sub-conclusions above their derivations

The final answer is kept unchanged (paper §2.1). Segment length, sampling, and the validation of
each rewrite are not specified by the paper; the choices made here, and every other deviation, are
in results/m4_part_design.md §4-§5.

Usage:
    python -m antidistill.defenses.part --input data/raw/limo_v2.json \
        --output data/defended/limo_part.json \
        --base-url http://127.0.0.1:8003/v1 --model Qwen/Qwen2.5-32B-Instruct
    ... --pilot 20     # the pre-run pilot (results/m4_part_design.md §7)
    ... --dry-run      # segment only; no server needed
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from antidistill.defenses.base import Trace

# --- verbatim from Ding et al. (2025), Appendix B ---------------------------------------------

PROMPT_REMOVE_SELF_TALK = (
    "Rewrite the given text, which is a part of a complete reasoning process. Convert only the "
    "parts expressed in a self-talk style into a declarative format. Avoid using first-person "
    "expressions such as 'I', 'me', 'we', or 'let's'. Do not alter any parts that are not "
    "self-talk; keep them exactly as in the original text.\n"
    "Do not add any extra information. Do not include any introductory phrases.\n"
    "Text:"
)

PROMPT_REORDER = (
    "You will process the given text in two steps. The given text is a part of a complete "
    "reasoning process.\n"
    "Step 1: Extract and list the most important sub-conclusions in the given reasoning process. "
    "Keep the number of sub-conclusions small and focused.\n"
    "Wrap the sub-conclusions in the tags <SUB> and </SUB> for easy extraction.\n"
    "Step 2: Move the sentences corresponding to these sub-conclusions to appear *before* their "
    "respective reasoning processes. Keep the sub-conclusions unnumbered and naturally integrated "
    "into the context. Do not modify any other parts of the original text.\n"
    "Wrap the entire transformed text in the tags <REWRITTEN> and </REWRITTEN> for easy "
    "extraction.\n"
    "Text:"
)

# -----------------------------------------------------------------------------------------------

FINAL_ANSWER = "**Final Answer**"

# Acceptable output/input length ratio per pass, checked only for inputs of >= MIN_RATIO_CHARS.
# Pass 1 may shrink a self-talk-heavy segment a lot, but not to a summary; pass 2 only moves text.
RATIO = {1: (0.35, 1.30), 2: (0.75, 1.35)}
MIN_RATIO_CHARS = 300
MISSING_NUMBERS_FRAC = 0.10      # of the input's distinct numbers, with at least 1 always allowed
# Minimum share of the input's distinct content words the output must keep. Unlike the ratio,
# this applies to segments of any length, so a short final segment cannot be replaced by
# unrelated text. Pass 1 rewrites the self-talk (whose words are excluded below) and may inflect
# the rest ("let me compute" -> "computing"); pass 2 only moves text.
CONTENT_RECALL = {1: 0.50, 2: 0.70}
MIN_CONTENT_WORDS = 5
_SELF_TALK_WORDS = {"wait", "maybe", "perhaps", "actually", "think", "okay", "alright", "let's",
                    "lets", "guess", "hmm", "hmmm", "well", "just", "mean", "seems", "sure",
                    "right", "really", "need", "should", "would", "could", "might", "going",
                    "want", "know", "remember", "recall", "check", "verify", "again", "also",
                    "first", "then", "next", "that", "this", "with", "from", "have", "what"}
_WORD = re.compile(r"[a-z]{4,}")


# ----------------------------------------------------------------------------- trace structure

def split_answer(output: str) -> tuple[str, str, str]:
    """(reasoning, gap, answer) with reasoning + gap + answer == output exactly.

    The final answer is kept verbatim (paper §2.1). LIMO ends 782/800 traces with a
    "**Final Answer**" section; 86 contain the marker more than once (a premature final answer,
    then more reasoning), so the split is at the LAST one. The 18 traces without it end on a
    sentence containing \\boxed{}: they are rewritten whole and their answer is checked instead.
    """
    k = output.rfind(FINAL_ANSWER)
    head, answer = (output[:k], output[k:]) if k != -1 else (output, "")
    reasoning = head.rstrip()
    return reasoning, head[len(reasoning):], answer


_SEP = re.compile(r"(\n[ \t]*\n\s*)")


def segment(text: str, max_tokens: int, count, min_tail: int | None = None) -> list[tuple[str, str]]:
    """Cut `text` at blank lines into paragraphs and pack consecutive paragraphs greedily into
    segments of at most `max_tokens` (by `count`, a batched token counter).

    A paragraph is never cut: one longer than `max_tokens` becomes a segment on its own. LIMO's
    longest paragraph is 3,711 characters, so in practice no segment is far over the target.
    A final segment shorter than `min_tail` (default max_tokens / 4) is merged into the one
    before it: greedy packing otherwise leaves stubs like "Therefore, the answer should be 64."
    on their own, and a stub that short cannot be validated (one number, two content words).

    Returns [(segment_text, separator_after)], and the split is exact:
        "".join(s + sep for s, sep in segments) == text
    """
    min_tail = max_tokens // 4 if min_tail is None else min_tail
    parts = _SEP.split(text)
    paras, seps = parts[0::2], parts[1::2] + [""]
    ntok = count([p + s for p, s in zip(paras, seps)])     # separators cost tokens too
    segs: list[list] = []                      # [text, separator_after, tokens]
    buf: list[str] = []                        # para, sep, para, sep, ... of the open segment
    buf_tok = 0
    for p, sep, n in zip(paras, seps, ntok):
        if buf and buf_tok + n > max_tokens:
            closing = buf.pop()                # separator after the open segment's last paragraph
            segs.append(["".join(buf), closing, buf_tok])
            buf, buf_tok = [], 0
        buf += [p, sep]
        buf_tok += n
    if buf:
        closing = buf.pop()
        segs.append(["".join(buf), closing, buf_tok])
    if len(segs) > 1 and segs[-1][2] < min_tail:
        tail = segs.pop()
        segs[-1] = [segs[-1][0] + segs[-1][1] + tail[0], tail[1], segs[-1][2] + tail[2]]
    return [(s, sep) for s, sep, _ in segs]


# ------------------------------------------------------------------------------ output parsing

_FENCE = re.compile(r"\A\s*```[a-zA-Z]*\n(.*?)\n```\s*\Z", re.S)
_PREAMBLE = re.compile(
    r"\A\s*(?:(?:sure|certainly|of course)[,.!]?\s*)?"
    r"(?:here(?:'s| is| are)|below is|the (?:rewritten|revised|converted|transformed) "
    r"(?:text|version)|rewritten text)[^\n]{0,120}?:[ \t]*\n+", re.I)


def clean_pass1(raw: str) -> tuple[str, list[str]]:
    """Pass-1 output with a code fence, an introductory line or an echoed "Text:" removed.

    The prompt forbids introductory phrases; when a model adds one anyway it is not part of the
    rewrite, and leaving it in would teach the student to open paragraphs with it.
    """
    t, flags = (raw or "").strip(), []
    m = _FENCE.match(t)
    if m:
        t, flags = m.group(1).strip(), flags + ["fence"]
    m = _PREAMBLE.match(t)
    if m:
        t, flags = t[m.end():].strip(), flags + ["preamble"]
    if t.startswith("Text:"):
        t, flags = t[len("Text:"):].strip(), flags + ["echoed_label"]
    return t, flags


_REWRITTEN = re.compile(r"<REWRITTEN>(.*?)</REWRITTEN>", re.S | re.I)
_SUB = re.compile(r"<SUB>(.*?)</SUB>", re.S | re.I)
_TAG = re.compile(r"</?(?:SUB|REWRITTEN)>", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def parse_pass2(raw: str, truncated: bool) -> tuple[str | None, list[str], list[str]]:
    """(rewritten text, sub-conclusions, flags) from a pass-2 response.

    The text is the last complete <REWRITTEN> block. If the closing tag is missing but the
    response finished normally, everything after the opening tag is taken and flagged; a
    truncated response yields None (retry). Sub-conclusions are the <SUB> items outside the
    rewritten block -- the model's step-1 list, split into items if it lists several in one
    block. Any <SUB>/<REWRITTEN> tags left inside the text are removed, keeping their content.
    """
    raw, flags = raw or "", []
    blocks = _REWRITTEN.findall(raw)
    if blocks:
        text = blocks[-1]
    else:
        k = raw.upper().rfind("<REWRITTEN>")
        if k == -1 or truncated:
            return None, [], ["no_rewritten_block"]
        text, flags = raw[k + len("<REWRITTEN>"):], ["unclosed"]
    subs: list[str] = []
    for b in _SUB.findall(_REWRITTEN.sub("", raw)):
        subs += [x for x in (_BULLET.sub("", ln).strip() for ln in b.strip().split("\n")) if x]
    # Qwen2.5-32B-Instruct usually marks the sub-conclusions INSIDE the rewritten text instead of
    # listing them first (264 of 269 pilot segments). Those spans are the hoisted sentences
    # themselves, so they count as sub-conclusions too.
    subs += [x for x in (s.strip() for s in _SUB.findall(text)) if x and x not in subs]
    if _TAG.search(text):
        text, flags = _TAG.sub("", text), flags + ["tags_in_text"]
    return text.strip(), subs, flags


# ---------------------------------------------------------------------------------------- guards

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    """Distinct numbers in `text` ("1,000" and "1000" are the same number)."""
    return set(_NUM.findall(_THOUSANDS.sub("", text or "")))


def missing_numbers(src: str, out: str) -> tuple[int, int]:
    """(distinct numbers of `src` absent from `out`, distinct numbers of `src`)."""
    s = numbers(src)
    return len(s - numbers(out)), len(s)


def content_words(text: str) -> set[str]:
    """Distinct lower-case words of 4+ letters, excluding self-talk and function words."""
    return set(_WORD.findall((text or "").lower())) - _SELF_TALK_WORDS


def content_recall(src: str, out: str) -> float | None:
    """Share of `src`'s content words present in `out`; None if `src` has too few to judge."""
    s = content_words(src)
    if len(s) < MIN_CONTENT_WORDS:
        return None
    return len(s & content_words(out)) / len(s)


_LIST_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s", re.M)
MAX_NEW_LIST_LINES = 2


def check(src: str, out: str, pass_no: int, finish: str | None) -> str | None:
    """None if `out` is an acceptable rewrite of `src`, otherwise the reason it is not.

    Cheap, deterministic, and aimed at the failures that would corrupt the training data:
    a truncated or empty response, a summary or unrelated text instead of a rewrite, text added,
    mathematics lost, and prose restructured into a list (the prompt says to keep everything that
    is not self-talk exactly as it was; a 1B smoke-test rewriter turned whole segments into
    numbered lists that kept every number). The paper's lexical similarity is measured too, but
    only for the audit: a segment that is mostly self-talk legitimately changes most sentences.
    """
    if finish == "length":
        return "truncated"
    if not out.strip():
        return "empty"
    if len(src) >= MIN_RATIO_CHARS:
        lo, hi = RATIO[pass_no]
        r = len(out) / len(src)
        if r < lo:
            return f"too_short:{r:.2f}"
        if r > hi:
            return f"too_long:{r:.2f}"
    miss, tot = missing_numbers(src, out)
    if miss > max(1, math.ceil(MISSING_NUMBERS_FRAC * tot)):
        return f"numbers_missing:{miss}/{tot}"
    rec = content_recall(src, out)
    if rec is not None and rec < CONTENT_RECALL[pass_no]:
        return f"content_lost:{rec:.2f}"
    added = len(_LIST_LINE.findall(out)) - len(_LIST_LINE.findall(src))
    if added > MAX_NEW_LIST_LINES:
        return f"list_introduced:{added}"
    return None


# ------------------------------------------------------------- information-preservation metrics

try:
    from rapidfuzz import fuzz as _fuzz

    def _align(short: str, long_: str) -> tuple[float, int]:
        r = _fuzz.partial_ratio_alignment(short, long_)
        return r.score, r.dest_start

    SIMILARITY_IMPL = "rapidfuzz.fuzz.partial_ratio_alignment"
except ImportError:                                              # pragma: no cover
    from difflib import SequenceMatcher

    def _align(short: str, long_: str) -> tuple[float, int]:
        """difflib version of the same alignment: best window of `long_` the length of `short`."""
        if not short:
            return 0.0, 0
        best, at = 0.0, 0
        for blk in SequenceMatcher(None, short, long_, autojunk=False).get_matching_blocks():
            start = max(0, blk.b - blk.a)
            r = SequenceMatcher(None, short, long_[start:start + len(short)], autojunk=False).ratio()
            if r > best:
                best, at = r, start
        return 100 * best, at

    SIMILARITY_IMPL = "difflib (rapidfuzz unavailable)"

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_UNIT_CHARS = 20


def _units(text: str, level: str) -> list[str]:
    raw = _SEP.split(text or "")[0::2] if level == "step" else _SENT.split(text or "")
    return [u.strip() for u in raw if len(u.strip()) >= MIN_UNIT_CHARS]


def lexical_match(original: str, rewritten: str, threshold: float = 0.7,
                  level: str = "sentence") -> tuple[int, int]:
    """The paper's lexical similarity (§3.1): (units of `original` found in `rewritten` with a
    partial_ratio_alignment score >= threshold, units). `level` is "sentence" or "step"
    (paragraph). Units under 20 characters are skipped: "Hmm." scored against a whole segment is
    noise, not evidence of preservation.
    """
    units = _units(original, level)
    return sum(_align(u, rewritten)[0] >= 100 * threshold for u in units), len(units)


def reorder_stats(before: str, after: str, subs: list[str]) -> dict:
    """Did pass 2 change the order? Two complementary measures:

    moved  -- sentences of `before` found verbatim in `after`, minus the longest run of them that
              kept its relative order (len - LIS): how many sentences changed position.
    hoisted -- sub-conclusions located (score >= 70) in both texts whose relative position moved
               earlier by at least 5% of the text. Catches a hoisted sentence the model reworded,
               which the verbatim search cannot see.
    """
    pos = [k for k in (after.find(u) for u in _units(before, "sentence")) if k != -1]
    tails: list[int] = []
    for p in pos:
        i = bisect.bisect_left(tails, p)
        tails[i:i + 1] = [p]
    hoisted = 0
    for s in subs:
        if len(s) < MIN_UNIT_CHARS or not before or not after:
            continue
        sb, pb = _align(s, before)
        sa, pa = _align(s, after)
        if sb >= 70 and sa >= 70 and pa / len(after) < pb / len(before) - 0.05:
            hoisted += 1
    return {"located": len(pos), "moved": len(pos) - len(tails), "subs": len(subs),
            "hoisted": hoisted}


_SENT_START = re.compile(r"(?:[.!?][\"')\]]?[ \t]+|\n[ \t]*)$")
_SENT_BOUNDARY = re.compile(r"[.!?][\"')\]]?[ \t]+|\n[ \t]*")
_LEAD_IN = re.compile(r"(?:[A-Za-z]+,?[ \t]+){1,4}")


def dedup_verbatim_copies(p1: str, p2: str, subs: list[str]) -> tuple[str, int, int]:
    """Undo pass 2's word-for-word copies, so each such sub-conclusion appears once, first.

    DEVIATION (antidistill), decided 2026-10-03 -- results/m4_part_design.md §7c. The paper's
    prompt asks the rewriter to MOVE each sub-conclusion ahead of its derivation. Ours usually
    COPIES it: 24% of sub-conclusions end up twice, word for word (the hoisted copy first, the
    original sentence still in place after its derivation). This removes the later copies.

    Only exact copies are touched; reworded summaries (another 33%) are left alone, since
    matching them to "their" original sentence would be a guess. A sub-conclusion counts as
    copied only if pass 2 holds it MORE often than pass 1 did -- LIMO sometimes repeats a sentence
    itself, and those repeats stay. The first occurrence (the hoisted one) is kept. A later
    occurrence is deleted only if it is a whole sentence: deleting text from inside a longer
    sentence would break it, so such cases are kept and counted.

    Returns (text, copies removed, copies left because they are not whole sentences).
    """
    removed = left = 0
    for s in dict.fromkeys(x.strip() for x in subs):          # unique, in order
        if len(s) < MIN_UNIT_CHARS or s not in p1:
            continue
        excess = p2.count(s) - p1.count(s)
        k = p2.find(s, p2.find(s) + len(s)) if excess > 0 else -1
        while excess > 0 and k != -1:
            end = k + len(s)
            if not re.search(r"[.!?]$", s) and end < len(p2) and p2[end] in ".!?":
                end += 1                                       # take the sentence's full stop
            start = k
            if k > 0 and _SENT_START.search(p2[:k]) is None:
                # The original often carries a short lead-in the hoisted copy dropped:
                # "So x can't be an integer." vs "x can't be an integer.". If what precedes it in
                # its sentence is at most four plain words (no digits, no math), the sentence is
                # still the original; remove it from its start.
                ss = max((m.end() for m in _SENT_BOUNDARY.finditer(p2, 0, k)), default=0)
                if _LEAD_IN.fullmatch(p2[ss:k]):
                    start = ss
            whole = ((start == 0 or _SENT_START.search(p2[:start]) is not None)
                     and re.search(r"[.!?][\"')\]]?$", p2[k:end]) is not None
                     and (end == len(p2) or p2[end].isspace()))
            if not whole:
                left += 1
                k = p2.find(s, end)
                continue
            k = start
            before, after = p2[:k], p2[end:].lstrip(" \t")
            if after.startswith("\n") or not after:
                before = before.rstrip(" \t")
            if before.endswith("\n") and after.startswith("\n"):
                after = after.lstrip("\n")                     # a removed one-sentence paragraph
            p2 = before + after
            removed += 1
            excess -= 1
            k = p2.find(s, k)
    return p2, removed, left


# -------------------------------------------------------------------------------------- engine

class _Client:
    """One or more OpenAI-compatible servers, each with a cap on requests in flight.

    Each request takes a slot on whichever server has the most free slots, and blocks while all
    are full. Round-robin would be wrong for unequal replicas (e.g. TP=4 next to TP=2): it sends
    each the same number of requests, so the slow one's queue grows without bound and sets the
    pace for the run. With caps, every server stays saturated at its own capacity and none
    queues. Transient failures are retried, releasing the slot while waiting.
    """

    def __init__(self, base_urls: list[str], model: str, timeout: float, api_retries: int,
                 caps: list[int] | None = None):
        from openai import OpenAI
        self.clients = [OpenAI(base_url=u, api_key="EMPTY", timeout=timeout, max_retries=0)
                        for u in base_urls]
        self.model, self.api_retries = model, api_retries
        self.free = list(caps) if caps else [1 << 30] * len(base_urls)
        assert len(self.free) == len(self.clients), "one cap per server"
        self._cv = threading.Condition()
        self.completion_tokens = 0
        self.served = [0] * len(self.clients)          # requests completed per server

    def _acquire(self) -> int:
        with self._cv:
            while max(self.free) <= 0:
                self._cv.wait()
            i = max(range(len(self.free)), key=self.free.__getitem__)
            self.free[i] -= 1
            return i

    def _release(self, i: int) -> None:
        with self._cv:
            self.free[i] += 1
            self._cv.notify()

    def chat(self, prompt: str, temperature: float, max_tokens: int) -> tuple[str, str | None]:
        last: Exception | None = None
        for attempt in range(self.api_retries + 1):
            i = self._acquire()
            try:
                r = self.clients[i].chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens)
                with self._cv:
                    self.completion_tokens += r.usage.completion_tokens if r.usage else 0
                    self.served[i] += 1
                return r.choices[0].message.content or "", r.choices[0].finish_reason
            except Exception as e:                      # timeout, connection reset, 5xx
                last = e
            finally:
                self._release(i)
            time.sleep(min(60, 5 * 2 ** attempt))
        raise last                                       # type: ignore[misc]


def _key(i: int, j: int, text: str) -> str:
    return f"{i}:{j}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


class PART:
    """The PART defense behind the `Defense` protocol (antidistill.defenses.base)."""

    name = "part"

    def __init__(self, base_urls: list[str] | None = None, model: str = "Qwen/Qwen2.5-32B-Instruct",
                 tokenizer: str = "Qwen/Qwen2.5-7B", segment_tokens: int = 1024,
                 max_retries: int = 3, retry_temperature: float = 0.7, max_new_tokens: int = 8192,
                 concurrency: int = 128, request_timeout: float = 1800.0, api_retries: int = 4,
                 cache_path: str | Path | None = None, server_caps: list[int] | None = None,
                 dedup_verbatim: bool = False):
        self.base_urls = base_urls or ["http://127.0.0.1:8003/v1"]
        # Per-server limits on requests in flight (one per base URL). With them, total
        # concurrency is their sum; without them, `concurrency` is shared out evenly.
        if server_caps is None:
            n = len(self.base_urls)
            server_caps = [concurrency // n + (k < concurrency % n) for k in range(n)]
        assert len(server_caps) == len(self.base_urls), "one --server-concurrency per --base-url"
        self.server_caps, concurrency = server_caps, sum(server_caps)
        self.model, self.tokenizer_name = model, tokenizer
        self.segment_tokens, self.max_retries = segment_tokens, max_retries
        self.retry_temperature, self.max_new_tokens = retry_temperature, max_new_tokens
        self.concurrency, self.request_timeout, self.api_retries = concurrency, request_timeout, api_retries
        self.cache_path = Path(cache_path) if cache_path else None
        self.dedup_verbatim = dedup_verbatim      # see dedup_verbatim_copies()
        self._tok = None
        self._client: _Client | None = None

    # -- Defense protocol --------------------------------------------------------------------

    def apply(self, traces: list[Trace]) -> list[Trace]:
        return self.run(traces)["defended"]

    def metadata(self) -> dict:
        return {
            "defense": self.name,
            "paper": "Ding, Cui, Dong, Zheng, Wei (2025), arXiv:2510.11545",
            "rewriter_model": self.model, "base_urls": self.base_urls,
            "server_caps": self.server_caps,
            "segment_tokens": self.segment_tokens, "tokenizer": self.tokenizer_name,
            "temperature": 0.0, "retry_temperature": self.retry_temperature,
            "max_retries": self.max_retries, "length_ratio_bounds": RATIO,
            "min_ratio_chars": MIN_RATIO_CHARS, "missing_numbers_frac": MISSING_NUMBERS_FRAC,
            "content_recall_min": CONTENT_RECALL, "min_content_words": MIN_CONTENT_WORDS,
            "prompts_sha1": hashlib.sha1((PROMPT_REMOVE_SELF_TALK + PROMPT_REORDER)
                                         .encode("utf-8")).hexdigest(),
            "similarity_impl": SIMILARITY_IMPL,
            "dedup_verbatim": self.dedup_verbatim,
        }

    # -- internals ---------------------------------------------------------------------------

    def count(self, texts: list[str]) -> list[int]:
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return [len(x) for x in self._tok(texts, add_special_tokens=False)["input_ids"]]

    def plan(self, traces: list[Trace], indices: list[int]) -> dict[int, dict]:
        """Split every selected trace into (reasoning segments, gap, answer). No API calls."""
        out = {}
        for i in indices:
            reasoning, gap, answer = split_answer(traces[i]["output"] or "")
            segs = segment(reasoning, self.segment_tokens, self.count)
            out[i] = {"segments": segs, "gap": gap, "answer": answer, "reasoning": reasoning,
                      "keys": [_key(i, j, s) for j, (s, _) in enumerate(segs)]}
        return out

    def _attempts(self, pass_no: int, src: str, n_tok: int, rec: dict) -> str | None:
        """Up to 1 + max_retries calls for one pass; the first acceptable output, else None."""
        prompt = (PROMPT_REMOVE_SELF_TALK if pass_no == 1 else PROMPT_REORDER) + "\n" + src
        scale = 1.5 if pass_no == 1 else 1.6           # pass 2 also emits the <SUB> list
        budget = min(self.max_new_tokens, int(scale * n_tok) + (256 if pass_no == 1 else 512))
        fails = rec[f"p{pass_no}_fail"]
        for attempt in range(self.max_retries + 1):
            temp = 0.0 if attempt == 0 else self.retry_temperature
            try:
                raw, finish = self._client.chat(prompt, temp, budget)
            except Exception as e:
                fails.append(f"api:{type(e).__name__}")
                continue
            if pass_no == 1:
                cand, flags = clean_pass1(raw)
            else:
                cand, subs, flags = parse_pass2(raw, finish == "length")
                if cand is None:
                    fails.append("truncated" if finish == "length" else flags[0])
                    if finish == "length":
                        budget = min(self.max_new_tokens, budget * 2)
                    continue
                # A parsed block is complete (an unclosed one only parses when the response
                # ended normally), so hitting the cap AFTER </REWRITTEN> truncated nothing used.
                finish = None
            why = check(src, cand, pass_no, finish)
            if why is None:
                rec["flags"] += [f"p{pass_no}_{f}" for f in flags]
                if pass_no == 2:
                    rec["subs"] = subs
                return cand
            fails.append(why)
            if why == "truncated":
                budget = min(self.max_new_tokens, budget * 2)
        return None

    def rewrite_segment(self, text: str) -> dict:
        """Both passes for one segment. Never raises: a pass that cannot produce an acceptable
        rewrite falls back (pass 1 -> the original text, pass 2 -> the pass-1 text) and is flagged,
        so no problem is ever dropped from the dataset and every fallback is countable."""
        rec = {"p1": text, "p2": text, "subs": [], "p1_fallback": False, "p2_fallback": False,
               "p1_fail": [], "p2_fail": [], "flags": []}
        if not text.strip():
            return rec
        n_tok = self.count([text])[0]
        p1 = self._attempts(1, text, n_tok, rec)
        if p1 is None:
            p1, rec["p1_fallback"] = text, True
        rec["p1"] = p1
        p2 = self._attempts(2, p1, self.count([p1])[0], rec)
        if p2 is None:
            p2, rec["p2_fallback"] = p1, True
        rec["p2"] = p2
        return rec

    def run(self, traces: list[Trace], indices: list[int] | None = None) -> dict:
        """Rewrite the selected traces (all by default), resuming from the segment cache.

        Returns {"defended", "step1", "meta", "indices"}: defended and step1 hold the selected
        records in index order with only "output" changed; meta has one audit row per trace.
        """
        indices = sorted(indices) if indices is not None else list(range(len(traces)))
        planned = self.plan(traces, indices)
        jobs = [(k, s) for i in indices for k, (s, _) in zip(planned[i]["keys"], planned[i]["segments"])]

        done: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            torn = 0
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:     # a kill mid-write leaves a partial line
                        torn += 1
                        continue
                    done[r["key"]] = r
            if torn:
                print(f"PART: skipped {torn} unreadable cache line(s); those segments are redone",
                      flush=True)
        pending = [(k, s) for k, s in jobs if k not in done]
        print(f"PART: {len(indices)} traces, {len(jobs)} segments, {len(jobs) - len(pending)} cached, "
              f"{len(pending)} to rewrite", flush=True)

        if pending:
            # A client may be injected beforehand (tests use a fake one); otherwise connect now.
            self._client = self._client or _Client(self.base_urls, self.model,
                                                   self.request_timeout, self.api_retries,
                                                   self.server_caps)
            lock, t0, n_done = threading.Lock(), time.time(), 0
            fh = open(self.cache_path, "a", encoding="utf-8") if self.cache_path else None
            every = max(1, len(pending) // 100)
            try:
                with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
                    futs = {ex.submit(self.rewrite_segment, s): k for k, s in pending}
                    for f in as_completed(futs):
                        rec = f.result()
                        rec["key"] = futs[f]
                        with lock:
                            done[rec["key"]] = rec
                            if fh:
                                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                fh.flush()
                            n_done += 1
                            if n_done % every == 0 or n_done == len(pending):
                                el = time.time() - t0
                                fb = sum(1 for k, _ in pending if k in done and
                                         (done[k]["p1_fallback"] or done[k]["p2_fallback"]))
                                eta = el / n_done * (len(pending) - n_done)
                                served = getattr(self._client, "served", None)
                                print(f"[{n_done}/{len(pending)}] fallbacks {fb} | "
                                      f"{self._client.completion_tokens / el:.0f} out-tok/s | "
                                      f"{'per-server ' + str(served) + ' | ' if served and len(served) > 1 else ''}"
                                      f"elapsed {el / 60:.1f} min | ETA {eta / 60:.1f} min", flush=True)
            finally:
                if fh:
                    fh.close()

        return self._assemble(traces, indices, planned, done)

    def _assemble(self, traces, indices, planned, done) -> dict:
        from antidistill.defenses.hindsight import answers_agree, load_grader
        math_equal, _ = load_grader()
        defended, step1, meta = [], [], []
        for i in indices:
            p = planned[i]
            recs = [done[k] for k in p["keys"]]
            seps = [sep for _, sep in p["segments"]]
            n_removed = n_left = 0
            if self.dedup_verbatim:
                fixed = []
                for r in recs:
                    t, a, b = dedup_verbatim_copies(r["p1"], r["p2"], r["subs"])
                    fixed.append({**r, "p2": t})
                    n_removed, n_left = n_removed + a, n_left + b
                recs = fixed
            new_reasoning = "".join(r["p2"].strip() + sep for r, sep in zip(recs, seps))
            s1_reasoning = "".join(r["p1"].strip() + sep for r, sep in zip(recs, seps))
            out = new_reasoning + p["gap"] + p["answer"]
            row = dict(traces[i]); row["output"] = out; defended.append(row)
            row1 = dict(traces[i]); row1["output"] = s1_reasoning + p["gap"] + p["answer"]
            step1.append(row1)

            lex_s = [lexical_match(s, r["p2"]) for (s, _), r in zip(p["segments"], recs)]
            lex_p = [lexical_match(s, r["p2"], level="step") for (s, _), r in zip(p["segments"], recs)]
            ro = [reorder_stats(r["p1"], r["p2"], r["subs"]) for r in recs]
            miss, tot = missing_numbers(p["reasoning"], new_reasoning)
            meta.append({
                "index": i,
                "segments": len(recs),
                "copies_removed": n_removed,
                "copies_left_inside_sentences": n_left,
                "p1_fallbacks": sum(r["p1_fallback"] for r in recs),
                "p2_fallbacks": sum(r["p2_fallback"] for r in recs),
                "segments_with_fallback": sum(1 for r in recs if r["p1_fallback"] or r["p2_fallback"]),
                "failed_attempts": sum(len(r["p1_fail"]) + len(r["p2_fail"]) for r in recs),
                "flags": sorted({f for r in recs for f in r["flags"]}),
                "has_final_answer_section": bool(p["answer"]),
                "reasoning_chars_in": len(p["reasoning"]),
                "reasoning_chars_out": len(new_reasoning),
                "numbers_missing": miss, "numbers_total": tot,
                "lexical_sentence": [sum(h for h, _ in lex_s), sum(n for _, n in lex_s)],
                "lexical_step": [sum(h for h, _ in lex_p), sum(n for _, n in lex_p)],
                "segments_reordered": sum(1 for x in ro if x["moved"] or x["hoisted"]),
                "sentences_moved": sum(x["moved"] for x in ro),
                "subs": sum(x["subs"] for x in ro),
                "subs_hoisted": sum(x["hoisted"] for x in ro),
                "answer_agrees": answers_agree(traces[i]["output"] or "", out, math_equal),
            })
        return {"defended": defended, "step1": step1, "meta": meta, "indices": indices}


# ----------------------------------------------------------------------------------------- CLI

def pilot_indices(traces: list[Trace], n: int, seed: int = 0) -> list[int]:
    """n traces: n/2 spread evenly across the length distribution, n/2 random (fixed seed)."""
    by_len = sorted(range(len(traces)), key=lambda i: len(traces[i]["output"] or ""))
    k = n // 2
    strat = {by_len[int((q + 0.5) * len(by_len) / k)] for q in range(k)}
    rest = [i for i in range(len(traces)) if i not in strat]
    return sorted(strat | set(random.Random(seed).sample(rest, n - len(strat))))


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/raw/limo_v2.json")
    ap.add_argument("--output", default="data/defended/limo_part.json")
    ap.add_argument("--base-url", action="append", default=None,
                    help="OpenAI-compatible endpoint; repeat for several replicas (round-robin)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B",
                    help="counts segment length; the student's tokenizer, so lengths mean training tokens")
    ap.add_argument("--segment-tokens", type=int, default=1024)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=128,
                    help="total requests in flight, split evenly across servers")
    ap.add_argument("--server-concurrency", default="",
                    help="comma-separated cap per --base-url, e.g. 224,32 for a TP=4 and a TP=2 "
                         "replica; overrides --concurrency")
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--api-retries", type=int, default=4)
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--pilot", type=int, default=0, help="rewrite only this many traces (§7 pilot)")
    sel.add_argument("--indices", default="", help="comma-separated trace indices")
    ap.add_argument("--dry-run", action="store_true", help="segment and report; no API calls")
    ap.add_argument("--dedup-verbatim", action="store_true",
                    help="remove pass 2's word-for-word copies of sub-conclusions (keep the first); "
                         "see dedup_verbatim_copies(). With a complete cache this makes no API calls")
    a = ap.parse_args()

    traces = json.loads(Path(a.input).read_text(encoding="utf-8"))
    if a.pilot:
        indices = pilot_indices(traces, a.pilot)
    elif a.indices:
        indices = [int(x) for x in a.indices.split(",")]
    else:
        indices = list(range(len(traces)))

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    part = PART(base_urls=a.base_url, model=a.model, tokenizer=a.tokenizer,
                segment_tokens=a.segment_tokens, max_retries=a.max_retries,
                retry_temperature=a.retry_temperature, max_new_tokens=a.max_new_tokens,
                concurrency=a.concurrency, request_timeout=a.request_timeout,
                api_retries=a.api_retries, cache_path=f"{stem}.segments.jsonl",
                server_caps=[int(x) for x in a.server_concurrency.split(",")] if a.server_concurrency else None,
                dedup_verbatim=a.dedup_verbatim)

    if a.dry_run:
        planned = part.plan(traces, indices)
        lens = sorted(n for p in planned.values() for n in part.count([s for s, _ in p["segments"]]))
        nseg = len(lens)
        print(f"{len(indices)} traces -> {nseg} segments (target {a.segment_tokens} tokens)")
        print(f"segment tokens: median {lens[nseg // 2]}, p95 {lens[int(.95 * nseg)]}, max {lens[-1]}")
        print(f"traces without a '{FINAL_ANSWER}' section: "
              f"{sum(1 for p in planned.values() if not p['answer'])}")
        return 0

    res = part.run(traces, indices)
    m = res["meta"]
    step1_path = Path(f"{stem}_step1.json")
    _write_json(out, res["defended"])
    _write_json(step1_path, res["step1"])
    _write_json(Path(f"{stem}.meta.json"),
                {"metadata": {**part.metadata(), "input": a.input, "indices": res["indices"],
                              "pilot": a.pilot, "written": time.strftime("%Y-%m-%d %H:%M:%S")},
                 "traces": m})

    seg = sum(r["segments"] for r in m)
    fb = sum(r["segments_with_fallback"] for r in m)
    ls = [sum(r["lexical_sentence"][k] for r in m) for k in (0, 1)]
    bad_ans = [r["index"] for r in m if r["answer_agrees"] is False]
    print(f"\nDONE: {len(m)} traces, {seg} segments -> {out}")
    print(f"  fallbacks          : {fb} segments ({100 * fb / max(seg, 1):.2f}%; gate <= 2%)")
    print(f"  lexical match @0.7 : {100 * ls[0] / max(ls[1], 1):.1f}% of sentences (gate >= 80%)")
    print(f"  answer disagrees   : {len(bad_ans)} {bad_ans[:10]}")
    print(f"  segments reordered : {sum(r['segments_reordered'] for r in m)}/{seg}")
    if a.dedup_verbatim:
        print(f"  copies removed     : {sum(r['copies_removed'] for r in m)} word-for-word "
              f"({sum(r['copies_left_inside_sentences'] for r in m)} left: inside longer sentences)")
    print(f"  step-1-only set    : {step1_path}")
    print("  audit              : scripts/36_part_audit.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
