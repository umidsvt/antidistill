"""Constants shared across defenses, scoring, and evaluation.

Sources are cited inline so every value can be traced back to the paper
(arXiv:2603.15500) or to Kim et al.'s released code.
"""

# Kim et al. §4.2: the nine surface tokens adopted as practical indicators of
# regions where epistemic verbalization occurs, with the GPT-5-judged
# co-occurrence frequency that motivated each one.
EPISTEMIC_TOKENS: dict[str, float] = {
    "wait": 0.730,
    "maybe": 0.329,
    "actually": 0.124,
    "check": 0.105,
    "hmm": 0.083,
    "perhaps": 0.082,
    "might": 0.066,
    "seems": 0.033,
    "alternatively": 0.012,
}

# Qwen tokenizer ids banned at inference in Kim et al.'s test-time suppression
# experiment (§6.1). Verbatim from eval/eval_suppressing_epistemic_verbalization.py.
# NOTE: these ids are Qwen-tokenizer specific. Resolve per-tokenizer with
# `tokenizer.encode(tok, add_special_tokens=False)` for any other model family.
EPISTEMIC_TOKEN_IDS_QWEN: list[int] = [
    14524,  # 'Wait'
    14144,  # ' Wait'
    3868,   # ' wait'
    81122,  # 'Hmm'
    89290,  # ' Hmm'
    8530,   # ' perhaps'
    32576,  # 'Perhaps'
    7344,   # ' maybe'
    10926,  # ' Maybe'
    22105,  # 'Maybe'
    93114,  # 'Alternatively'
    39578,  # ' Alternatively'
    5084,   # ' seems'
    2643,   # ' might'
    4461,   # ' likely'
    8101,   # ' guess'
    2771,   # ' sure'
    2500,   # ' another'
    14364,  # 'Another'
]

# Appendix H.1: substitutions the models used to route around the banned
# vocabulary. Epistemic behaviour that our density metric should still catch,
# so a lexical-only counter does not silently score a defended trace at zero.
EPISTEMIC_BYPASS_PHRASES: list[str] = [
    "hold on",
    "let me verify",
    "let me double-check",
    "that's not quite right",
    "it's possible that",
    "another way to see this",
    "i realize",
    "on closer look",
    "is that correct",
    "let me reconsider",
    "i'm not sure",
]

# System prompt used by both LIMO training data and the eval harness
# (make_limo_dataset.py, eval/prompts/qwen-instruct/*.py).
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
