#!/usr/bin/env python
"""Examples for reading (results/phase0_examples.md): a few test problems of GSM8K, MATH and the puzzle testbed, with
the teachers' solutions and the students' attempts, side by side. Built from the stored per-problem files of setup 2
(§12), Phase 0 (§24) and the puzzle testbed (§18, §24 addendum). Sources not yet written are listed as pending, so
the file can be rebuilt as runs finish.

    PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_OFFLINE=1 .venv-dg/bin/python scripts/phase0_examples.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/phase0_examples.md"
HEAD, TAIL = 900, 600   # characters kept from the start and the end of a long solution


def trim(text: str) -> str:
    text = text.strip()
    if len(text) <= HEAD + TAIL + 200:
        return text
    return f"{text[:HEAD]}\n\n[… {len(text) - HEAD - TAIL:,} characters omitted …]\n\n{text[-TAIL:]}"


def block(title: str, text: str | None, meta: str) -> str:
    if text is None:
        return f"**{title}:** pending ({meta})\n"
    return f"**{title}** ({meta})\n\n````text\n{trim(text)}\n````\n"


def load_json(path: str):
    files = glob.glob(str(REPO / path))
    return json.loads(Path(files[0]).read_text()) if files else None


def rows_by_id(path: str) -> dict | None:
    d = load_json(path)
    if d is None:
        return None
    rows = d["rows"] if isinstance(d, dict) else d
    return {r["example_id"]: r for r in rows}


def as_bool(v) -> bool:
    return v is True or v == "True"


def setup2_section(ds: str, picks_wanted: int) -> str:
    from clean_sweep.config import FullConfig
    from clean_sweep.data import load_dataset_splits
    cfg = FullConfig.from_yaml(str(REPO / f"configs/setup2/{ds}_U_s123.yaml"))
    test = load_dataset_splits(ds, seed=123, train_size=cfg.data.train_size, holdout_size=cfg.data.holdout_size,
                               test_size=cfg.data.test_size)["test"]
    gold = {test[i]["example_id"]: (test[i]["problem"], test[i]["solution"]) for i in range(len(test))}
    field = rows_by_id(f"outputs/setup2/{ds}_U_s123_*/teacher/test_standard.json")
    big = rows_by_id(f"outputs/phase0/teacher32b_{ds}/test.json")
    i0 = rows_by_id(f"outputs/phase0/I0/{ds}_s123_1024.json")
    s1 = rows_by_id(f"outputs/phase0/S1/{ds}_s123/test_1024.json")
    s2 = rows_by_id(f"outputs/phase0/S2/{ds}_s123/test_8192.json")

    def ok(rows, eid):
        return rows is not None and as_bool(rows[eid]["correct"])

    # informative cases, shortest problems first: (1) the teacher right, the untrained student wrong; (2) the
    # untrained student right, the field-trained student wrong; (3) everyone with a file right
    order = sorted(gold, key=lambda e: len(gold[e][0]))
    cases = [
        ("the teacher is right, the untrained student is wrong",
         lambda e: ok(field, e) and (big is None or ok(big, e)) and not ok(i0, e)),
        ("the untrained student is right, the student trained on the field teacher is wrong",
         lambda e: ok(i0, e) and s1 is not None and not ok(s1, e)),
        ("all right", lambda e: ok(field, e) and ok(i0, e) and (big is None or ok(big, e)) and (s1 is None or ok(s1, e))),
    ]
    picks = []
    for why, cond in cases:
        e = next((e for e in order if e not in [p for p, _ in picks] and cond(e)), None)
        if e is not None:
            picks.append((e, why))
    picks = picks[:picks_wanted]
    out = []
    for k, (eid, why) in enumerate(picks, 1):
        problem, solution = gold[eid]
        out.append(f"### {ds.upper()} example {k} ({eid}): {why}\n\n**Problem.** {problem.strip()}\n\n"
                   f"**Reference answer:** {solution.strip()[-300:]}\n")

        def teacher_text(rows):
            if rows is None:
                return None, "file not written yet"
            r = rows[eid]
            text = (r.get("trace") or "") + "\n\n" + (r.get("af_final_answer_only") or "")
            meta = f"{'correct' if as_bool(r['correct']) else 'wrong'}, answer {r.get('extracted_answer')}"
            if "n_tokens" in r:
                meta += f", {r['n_tokens']} tokens, {'finished' if r.get('finished') else 'cut at the limit'}"
            return text, meta

        def student_text(rows):
            if rows is None:
                return None, "run not finished yet"
            r = rows[eid]
            return r["text"], (f"{'correct' if r['correct'] else 'wrong'}, answer {r['extracted_answer']}, "
                               f"{r['n_tokens']} tokens, {'finished' if r['finished'] else 'cut at the limit'}")
        for title, (text, meta) in [
            ("Field teacher, R1-Distill-Qwen-7B, 1,024 tokens (the setup's own run)", teacher_text(field)),
            ("Bigger teacher, R1-Distill-Qwen-32B, 8,192 tokens", teacher_text(big)),
            ("Untrained student, Llama-3.2-3B-Instruct, 1,024 tokens", student_text(i0)),
            ("Student trained on the field teacher's solutions (S1), 1,024 tokens", student_text(s1)),
            ("Student trained on the 32B teacher's solutions (S2), 8,192 tokens", student_text(s2)),
        ]:
            out.append(block(title, text, meta))
        out.append("Each scored text is followed, in scoring, by the setup's forced answer cue; the answer shown is the "
                   "one the checker read.\n")
    return "\n".join(out)


def kk_section() -> str:
    from kk_pilot import parse_labels
    test = {json.loads(line)["id"]: json.loads(line) for line in open(REPO / "data/kk/step2_test.jsonl")}
    six = [json.loads(line)["id"] for line in open(REPO / "outputs/kk_step2/examples/test6.jsonl")] \
        if (REPO / "outputs/kk_step2/examples/test6.jsonl").exists() else []
    if not six:
        ids7 = [i for i, r in test.items() if r["n_people"] == 7][:3]
        ids8 = [i for i, r in test.items() if r["n_people"] == 8][:3]
        six = ids7 + ids8
    picks = [six[0], six[3]]   # one puzzle with 7 people, one with 8

    def tries(path):
        p = REPO / path
        if not p.exists():
            return None
        return {r["id"]: r for r in map(json.loads, open(p))}
    teacher = tries("outputs/kk_step2/examples/T_test/tries.jsonl")
    sources = [
        ("Teacher, R1-Distill-Qwen-32B (8,192 tokens)", teacher, True),
        ("Untrained base student, Llama-3.2-3B (§18)", tries("outputs/kk_step2/students/untrained_s123/test_eot/results.jsonl"), False),
        ("Base student trained on the teacher's solutions, U (§18)", tries("outputs/kk_step2/students/U_s123/test_eot/results.jsonl"), False),
        ("Untrained instruct student, Llama-3.2-3B-Instruct", tries("outputs/kk_step2/students/untrained_inst_s123/test/results.jsonl"), False),
        ("Instruct student trained on the teacher's solutions, U_inst", tries("outputs/kk_step2/students/Uinst_s123/test/results.jsonl"), False),
    ]
    out = []
    for k, pid in enumerate(picks, 1):
        r = test[pid]
        truth = ", ".join(f"{n} is a {'knight' if v else 'knave'}" for n, v in zip(r["names"], r["solution"]))
        out.append(f"### Puzzle example {k} ({pid}, {r['n_people']} people)\n\n**Puzzle.** {r['quiz'].strip()}\n\n"
                   f"**Solution:** {truth}.\n")
        for title, rows, is_teacher in sources:
            if rows is None or pid not in rows:
                out.append(block(title, None, "run not finished yet"))
                continue
            t = rows[pid]["tries"][0] if is_teacher else rows[pid]
            text = t["text"]
            labels = parse_labels(text, r["names"], True) if is_teacher else t.get("labels")
            right = labels == list(r["solution"])
            meta = (f"{'correct' if right else ('wrong' if labels is not None else 'no readable answer')}, "
                    f"{t['tokens']} tokens, {'finished' if t['finish_reason'] == 'stop' else 'cut at the limit'}")
            out.append(block(title, text, meta))
    return "\n".join(out)


def main() -> int:
    parts = ["# Phase 0 examples: how the teachers solve test problems and how the students try",
             "",
             "Built by `scripts/phase0_examples.py` from the stored per-problem files (m8 §12, §18, §24); rebuilt as "
             "runs finish. Seed 123 throughout. Long solutions show their first 900 and last 600 characters. Problems "
             "are chosen automatically, shortest first, to show typical cases: the teacher right and the untrained "
             "student wrong; the untrained student right and the trained student wrong; everyone right.",
             "",
             "## GSM8K (GSM8K-Platinum test)", "", setup2_section("gsm8k", 2),
             "## MATH (test)", "", setup2_section("math", 3),
             "## Knights and Knaves (step-2 test puzzles)", "",
             "Students answer in the format they were trained on: reasoning, then a CONCLUSION list. The base students "
             "cannot end their turn (m8 §18), so their text runs on after the answer; their first answer is scored.",
             "", kk_section()]
    OUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
