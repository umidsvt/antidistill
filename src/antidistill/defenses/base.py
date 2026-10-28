"""The contract every trace-corruption defense implements (IMPLEMENTATION_PLAN.md §6).

A defense takes the teacher's training records and returns defended ones, one for one and in the
same order, so a defended dataset always covers exactly the problems of its source -- the
"same 800 problems, only the traces differ" design that every contrast in this study rests on.

Records are LLaMA-Factory alpaca rows: {"instruction", "input", "output", "system"}. A defense
rewrites "output" and leaves the other fields alone.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

Trace = dict


@runtime_checkable
class Defense(Protocol):
    name: str

    def apply(self, traces: list[Trace]) -> list[Trace]:
        """Defended copies of `traces`: same length, same order, only "output" changed."""
        ...

    def metadata(self) -> dict:
        """Parameters, model versions and timing -- enough to reproduce the run."""
        ...
