"""Offline checks for the PART defense (src/antidistill/defenses/part.py). No server, no GPU.

Run from antidistill/ with either:
    .venv-infer/bin/python tests/test_part.py
    .venv-infer/bin/python -m pytest tests/test_part.py      # if pytest is installed

The structural checks run on the real LIMO-v2 data (all 800 traces), because the guarantees that
matter -- an exact segment round-trip, an untouched final answer -- are properties of that data.
The end-to-end checks drive the full pipeline with a fake rewriter in place of the LLM.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from antidistill.defenses.base import Defense  # noqa: E402
from antidistill.defenses.part import (  # noqa: E402
    FINAL_ANSWER, PART, PROMPT_REMOVE_SELF_TALK, PROMPT_REORDER, check, clean_pass1,
    lexical_match, missing_numbers, numbers, parse_pass2, pilot_indices, reorder_stats,
    segment, split_answer)

LIMO = REPO / "data" / "raw" / "limo_v2.json"
_TRACES = None
_PART = None


def traces():
    global _TRACES
    if _TRACES is None:
        _TRACES = json.loads(LIMO.read_text(encoding="utf-8"))
    return _TRACES


def part() -> PART:
    global _PART
    if _PART is None:
        _PART = PART()
    return _PART


# ------------------------------------------------------------------------ structure, real data

def test_split_answer_is_exact_on_all_limo_traces():
    with_answer = 0
    for r in traces():
        reasoning, gap, answer = split_answer(r["output"])
        assert reasoning + gap + answer == r["output"]
        assert reasoning == reasoning.rstrip()
        if answer:
            with_answer += 1
            assert answer.startswith(FINAL_ANSWER)
            assert FINAL_ANSWER not in answer[len(FINAL_ANSWER):]      # split at the LAST marker
    assert with_answer == 782, with_answer


def test_segment_round_trip_is_exact_on_all_limo_traces():
    p = part()
    over = 0
    for r in traces():
        reasoning, _, _ = split_answer(r["output"])
        segs = segment(reasoning, 1024, p.count)
        assert "".join(s + sep for s, sep in segs) == reasoning
        assert segs[-1][1] == ""
        for j, (s, _) in enumerate(segs):
            n = p.count([s])[0]
            # the last segment may carry a merged tail of up to max_tokens/4
            limit = 1024 + (256 if j == len(segs) - 1 else 0)
            if n > limit:
                over += 1
                # only a single paragraph that is itself over target, or BPE merging across a
                # paragraph boundary differently from the per-paragraph count (a few tokens)
                assert n <= limit * 1.02 or not re.search(r"\n[ \t]*\n", s), (n, s[:80])
            if len(segs) > 1 and j == len(segs) - 1:
                assert n >= 256 * 0.95, n               # no stub tails left
    assert over < 100, over


def test_segment_small_cases():
    count = lambda xs: [len(x.split()) for x in xs]     # noqa: E731  (1 token per word)
    t = "a b\n\nc d e\n\n\nf"
    segs = segment(t, 3, count)
    assert "".join(s + sep for s, sep in segs) == t
    assert [s for s, _ in segs] == ["a b", "c d e", "f"]          # min_tail 0 by default here
    assert segment("one para", 3, count) == [("one para", "")]
    assert segment("", 3, count) == [("", "")]
    # a short tail is folded into the previous segment, keeping the split exact
    segs = segment(t, 3, count, min_tail=2)
    assert [s for s, _ in segs] == ["a b", "c d e\n\n\nf"] and segs[-1][1] == ""
    assert "".join(s + sep for s, sep in segs) == t


def test_pilot_selection_is_deterministic_and_sized():
    a, b = pilot_indices(traces(), 20), pilot_indices(traces(), 20)
    assert a == b and len(a) == 20 and a == sorted(a)


# ------------------------------------------------------------------------------------ parsing

def test_clean_pass1():
    assert clean_pass1("Here is the rewritten text:\n\nx = 4.") == ("x = 4.", ["preamble"])
    assert clean_pass1("Sure! Here's the text with self-talk converted:\nx = 4.")[0] == "x = 4."
    assert clean_pass1("```\nx = 4.\n```") == ("x = 4.", ["fence"])
    assert clean_pass1("Text: x = 4.") == ("x = 4.", ["echoed_label"])
    plain = "Here we compute x: it is 4.\nSo x = 4."
    assert clean_pass1(plain) == (plain, [])        # a colon mid-sentence is not a preamble


def test_parse_pass2():
    raw = ("<SUB>x = 4</SUB>\n<SUB>- y = 12\n- z = 3</SUB>\n"
           "<REWRITTEN>So x = 4. Since 2x = 8, x = 4.</REWRITTEN>")
    text, subs, flags = parse_pass2(raw, truncated=False)
    assert text == "So x = 4. Since 2x = 8, x = 4."
    assert subs == ["x = 4", "y = 12", "z = 3"] and flags == []
    text, _, flags = parse_pass2("<SUB>a</SUB><REWRITTEN>body without close", truncated=False)
    assert text == "body without close" and flags == ["unclosed"]
    assert parse_pass2("<REWRITTEN>body without close", truncated=True)[0] is None
    assert parse_pass2("no tags at all", truncated=False)[0] is None
    text, subs, flags = parse_pass2("<REWRITTEN><SUB>x = 4</SUB> because 2x = 8</REWRITTEN>", False)
    assert text == "x = 4 because 2x = 8" and flags == ["tags_in_text"]
    assert subs == ["x = 4"]                       # inline-marked sub-conclusions are counted


# ------------------------------------------------------------------------------------- guards

def test_numbers():
    assert numbers("1,000 and 1000 and 29.98, then 3.") == {"1000", "29.98", "3"}
    assert missing_numbers("x = 12, y = 7", "x = 12") == (1, 2)


SRC = ("Hmm, let me think about the rectangle. Its length is 12 and its width is 5, so the "
       "perimeter is 2 * (12 + 5) = 34. Wait, I should double-check the area as well: "
       "12 * 5 = 60. Okay, so the diagonal follows from Pythagoras: sqrt(144 + 25) = 13. "
       "Let me verify that 13 squared is 169, which matches 144 + 25.")
GOOD = ("Consider the rectangle. Its length is 12 and its width is 5, so the perimeter is "
        "2 * (12 + 5) = 34. The area is 12 * 5 = 60. The diagonal follows from Pythagoras: "
        "sqrt(144 + 25) = 13. Indeed 13 squared is 169, which matches 144 + 25.")


def test_check():
    assert check(SRC, GOOD, 1, "stop") is None
    assert check(SRC, GOOD, 1, "length") == "truncated"
    assert check(SRC, "  ", 1, "stop") == "empty"
    assert check(SRC, "The diagonal is 13 = sqrt(144 + 25), area 60 = 12 * 5.", 1, "stop").startswith("too_short")
    assert check(SRC, SRC * 2, 1, "stop").startswith("too_long")
    no_nums = re.sub(r"\d+", "N", GOOD)
    assert check(SRC, no_nums, 1, "stop").startswith("numbers_missing")
    # pass 2 is stricter on shrinking: reordering moves text, it does not remove it
    long_ = GOOD + " " + GOOD                                # over the 300-char ratio threshold
    cut = long_[: int(len(long_) * 0.6)] + " 12 5 34 60 144 25 13 169 2"
    assert check(long_, cut, 1, "stop") is None              # fine for pass 1 (>= 0.35)...
    assert check(long_, cut, 2, None).startswith("too_short")   # ...not for pass 2 (< 0.75)
    # below the ratio threshold, content recall still rejects unrelated text
    short = "The rectangle has length 12 and width 5, so its perimeter equals 34 units."
    assert check(short, "Here is a short summary.", 1, "stop").startswith("numbers_missing")
    assert check(short, "Here is a short summary of 12, 5 and 34.", 1, "stop").startswith("content_lost")
    assert check(short, "The rectangle has length 12 and width 5; its perimeter equals 34 units.",
                 1, "stop") is None
    # prose restructured into a list keeps every number and word, and is still not a rewrite
    listed = "\n".join(f"{k}. {s}" for k, s in enumerate(re.split(r"(?<=\.) ", GOOD), 1))
    assert check(SRC, listed, 1, "stop").startswith("list_introduced")


def test_lexical_match_and_reorder():
    a = "Expanding the binomial gives sixteen terms in total."
    b = "Therefore the probability of drawing a red ball is three eighths."
    orig = f"{a} {b}"
    assert lexical_match(orig, orig) == (2, 2)
    assert lexical_match(orig, b) == (1, 2)
    # The paper's partial_ratio is lenient: two sentences sharing a frame score >= 70 against
    # each other. That is the metric as published; the audit reports it, it is not a guard.
    assert lexical_match("The value of x equals four.", "The value of y equals twelve.")[0] == 1
    st = reorder_stats(orig, f"{b} {a}", ["the probability of drawing a red ball is three eighths"])
    assert st["moved"] == 1 and st["hoisted"] == 1
    st = reorder_stats(orig, orig, [])
    assert st["moved"] == 0 and st["hoisted"] == 0


def test_dedup_verbatim_copies():
    from antidistill.defenses.part import dedup_verbatim_copies
    sub = "Therefore, the new total age is 160."
    p1 = "A person leaves, so the total is 130. Adding 30 gives 160. " + sub
    # hoisted copy first, original still at the end -> the original goes, the copy stays
    p2 = sub + " A person leaves, so the total is 130. Adding 30 gives 160. " + sub
    out, removed, left = dedup_verbatim_copies(p1, p2, [sub])
    assert out == sub + " A person leaves, so the total is 130. Adding 30 gives 160."
    assert (removed, left) == (1, 0)
    # a one-sentence paragraph removed cleanly, without leaving a blank gap
    p2b = sub + "\n\nA person leaves, so the total is 130.\n\n" + sub + "\n\nNext step."
    out, removed, _ = dedup_verbatim_copies(p1, p2b, [sub])
    assert out == sub + "\n\nA person leaves, so the total is 130.\n\nNext step." and removed == 1
    # the later occurrence sits inside a longer sentence -> left alone, counted
    frag = "the new total age is 160"
    p1c = "Adding 30, " + frag + " years. Done here."
    p2c = frag + ". Adding 30, " + frag + " years. Done here."
    out, removed, left = dedup_verbatim_copies(p1c, p2c, [frag])
    assert out == p2c and (removed, left) == (0, 1)
    # the original keeps a short lead-in the hoisted copy dropped -> the whole original goes
    s2 = "x can't be an integer."
    p1s = "Then x * x = 210, which is not a square. So " + s2 + " Next."
    p2s = s2 + " Then x * x = 210, which is not a square. So " + s2 + " Next."
    out, removed, left = dedup_verbatim_copies(p1s, p2s, [s2])
    assert out == s2 + " Then x * x = 210, which is not a square. Next." and (removed, left) == (1, 0)
    s3 = "the largest number so far is 876321."
    p2t = s3 + " Earlier attempts failed. So, going back, " + s3 + " Done."
    out, removed, _ = dedup_verbatim_copies("So, going back, " + s3 + " Done.", p2t, [s3])
    assert out == s3 + " Earlier attempts failed. Done." and removed == 1
    # a lead-in holding numbers or math is content, not a connective -> left alone
    p2m = s2 + " Since 210 = 14 * 15 " + s2 + " Next."
    out, removed, left = dedup_verbatim_copies("Since 210 = 14 * 15 " + s2 + " Next.", p2m, [s2])
    assert out == p2m and (removed, left) == (0, 1)
    # LIMO already repeated the sentence: only the rewriter's extra copy goes
    p1d = sub + " Checking again. " + sub
    p2d = sub + " " + sub + " Checking again. " + sub
    out, removed, _ = dedup_verbatim_copies(p1d, p2d, [sub])
    assert out.count(sub) == 2 and removed == 1
    # not in pass 1 at all (a reworded summary) -> untouched
    p2e = "The total is 160 now. " + p1
    assert dedup_verbatim_copies(p1, p2e, ["The total is 160 now."]) == (p2e, 0, 0)


def test_prompts_are_the_papers():
    # Appendix B of arXiv:2510.11545, verbatim. Changing them changes the defense.
    assert PROMPT_REMOVE_SELF_TALK.startswith("Rewrite the given text, which is a part of a complete")
    assert "'I', 'me', 'we', or 'let's'" in PROMPT_REMOVE_SELF_TALK
    assert PROMPT_REORDER.count("<SUB>") == 1 and PROMPT_REORDER.count("<REWRITTEN>") == 1
    assert "*before*" in PROMPT_REORDER
    assert PROMPT_REMOVE_SELF_TALK.endswith("Text:") and PROMPT_REORDER.endswith("Text:")


def test_part_implements_defense_protocol():
    assert isinstance(PART(), Defense)


def test_dispatch_respects_per_server_caps():
    """A TP=4 and a TP=2 replica: each stays within its cap, and the bigger one does more work."""
    import threading
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from antidistill.defenses.part import _Client

    class FakeServer:
        def __init__(self, delay):
            self.delay, self.now, self.peak, self.lock = delay, 0, 0, threading.Lock()
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            with self.lock:
                self.now += 1
                self.peak = max(self.peak, self.now)
            _time.sleep(self.delay)
            with self.lock:
                self.now -= 1
            msg = SimpleNamespace(content="ok")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                                   usage=SimpleNamespace(completion_tokens=1))

    c = _Client(["http://a/v1", "http://b/v1"], "m", 10, 0, caps=[6, 2])
    c.clients = [FakeServer(0.01), FakeServer(0.03)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: c.chat("p", 0.0, 4), range(200)))
    assert c.clients[0].peak <= 6 and c.clients[1].peak <= 2
    assert sum(c.served) == 200 and c.served[0] > 3 * c.served[1], c.served
    assert c.free == [6, 2]                                  # every slot returned


def test_concurrency_is_split_or_capped():
    assert PART(base_urls=["a", "b"], concurrency=129).server_caps == [65, 64]
    p = PART(base_urls=["a", "b"], server_caps=[224, 32])
    assert p.concurrency == 256


# -------------------------------------------------------------------- end to end, fake rewriter

class FakeRewriter:
    """Stands in for the LLM. Pass 1 deletes "Hmm, " / "Wait, " (a crude self-talk removal);
    pass 2 moves the last sentence to the front and reports it as the sub-conclusion."""

    def __init__(self, break_pass1: bool = False):
        self.calls, self.completion_tokens, self.break_pass1 = 0, 0, break_pass1

    def chat(self, prompt, temperature, max_tokens):
        self.calls += 1
        text = prompt.split("\nText:\n", 1)[1]
        if prompt.startswith(PROMPT_REMOVE_SELF_TALK):
            if self.break_pass1:
                return "Here is a short summary.", "stop"
            return re.sub(r"\b(Hmm|Wait), ", "", text), "stop"
        sents = re.split(r"(?<=[.!?]) ", text)
        moved = " ".join([sents[-1]] + sents[:-1]) if len(sents) > 1 else text
        return f"<SUB>{sents[-1]}</SUB>\n<REWRITTEN>{moved}</REWRITTEN>", "stop"


def _run_fake(indices, fake, tmp):
    p = PART(cache_path=Path(tmp) / "cache.jsonl", concurrency=4)
    p._client = fake
    return p.run(traces(), indices)


def test_end_to_end_with_fake_rewriter_and_resume():
    idx = [0, 39, 400]                       # 39 has no "**Final Answer**" section
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeRewriter()
        res = _run_fake(idx, fake, tmp)
        assert [r["instruction"] for r in res["defended"]] == [traces()[i]["instruction"] for i in idx]
        for i, d, s1, m in zip(idx, res["defended"], res["step1"], res["meta"]):
            orig = traces()[i]["output"]
            _, _, answer = split_answer(orig)
            assert d["output"].endswith(answer) and s1["output"].endswith(answer)
            assert d["output"] != orig
            assert m["p1_fallbacks"] == 0 and m["p2_fallbacks"] == 0
            assert m["answer_agrees"] in (True, None)
            assert m["numbers_missing"] == 0
        assert res["meta"][1]["has_final_answer_section"] is False
        n_calls = fake.calls
        assert n_calls == 2 * sum(m["segments"] for m in res["meta"])
        # resume: everything is cached, so a second run makes no calls and gives the same data
        fake2 = FakeRewriter()
        res2 = _run_fake(idx, fake2, tmp)
        assert fake2.calls == 0
        assert [r["output"] for r in res2["defended"]] == [r["output"] for r in res["defended"]]


def test_fallback_keeps_the_original_and_is_counted():
    with tempfile.TemporaryDirectory() as tmp:
        res = _run_fake([0], FakeRewriter(break_pass1=True), tmp)
        m = res["meta"][0]
        assert m["p1_fallbacks"] == m["segments"]          # every pass 1 was rejected...
        assert res["step1"][0]["output"] == traces()[0]["output"]   # ...so step 1 is the original
        assert m["failed_attempts"] >= 4 * m["segments"]    # 1 try + 3 retries each


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS  {n}")
        except Exception as e:                               # report all, not just the first
            failed += 1
            print(f"FAIL  {n}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
