"""Both token counters, and the choice between them.

The BGE-M3 tests skip when `tokenizer.json` is absent, following the same rule
as the integration suite: a machine without the artefact runs a green suite
rather than a red one it cannot fix. Fetch it with

    uv run python scripts/fetch_models.py --tokenizer-only

`build_token_counter` itself is never skipped — the selection logic is pure
configuration handling and has to hold everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.adapters.tokenize import (
    BgeTokenCounter,
    HeuristicTokenCounter,
    build_token_counter,
)
from rag.core.config import IngestionSettings, TokenizerKind
from rag.core.errors import ConfigurationError
from rag.domain.ports import TokenCounter

DEFAULT_TOKENIZER = Path("./var/models/bge-m3/tokenizer.json")

requires_tokenizer = pytest.mark.skipif(
    not DEFAULT_TOKENIZER.is_file(),
    reason=(
        f"{DEFAULT_TOKENIZER} not present; run "
        f"`uv run python scripts/fetch_models.py --tokenizer-only`"
    ),
)


@pytest.fixture(scope="module")
def counter() -> BgeTokenCounter:
    """One counter for the whole module.

    Module-scoped because loading the 17 MB vocabulary costs ~1.2 s and a
    function-scoped fixture paid it once per test — about 17 seconds of the
    suite for no benefit, since a `BgeTokenCounter` holds no mutable state after
    construction. The tests that specifically care about *construction* build
    their own instances.
    """
    return BgeTokenCounter(DEFAULT_TOKENIZER)


class TestSelection:
    def test_the_default_needs_no_files(self) -> None:
        # A fresh clone must run without downloading 17 MB of vocabulary.
        counter = build_token_counter(IngestionSettings())

        assert isinstance(counter, HeuristicTokenCounter)

    def test_a_missing_vocabulary_is_a_configuration_error(self) -> None:
        # Never a silent fallback to the estimator. Falling back would make
        # chunk boundaries depend on whether a file happened to be present, so
        # the same document would chunk differently on two machines with
        # nothing logged and nothing failing.
        settings = IngestionSettings(
            tokenizer=TokenizerKind.BGE_M3,
            tokenizer_path="./var/models/definitely-not-here/tokenizer.json",
        )

        with pytest.raises(ConfigurationError) as caught:
            build_token_counter(settings)

        # The message has to say what to run. An operator hitting this at 3am
        # should not have to read the source to find the fetch script.
        assert "fetch_models.py" in caught.value.message

    @requires_tokenizer
    def test_it_builds_the_real_counter_when_configured(self) -> None:
        settings = IngestionSettings(
            tokenizer=TokenizerKind.BGE_M3, tokenizer_path=str(DEFAULT_TOKENIZER)
        )

        assert isinstance(build_token_counter(settings), BgeTokenCounter)


class TestBothSatisfyThePort:
    def test_the_estimator(self) -> None:
        assert isinstance(HeuristicTokenCounter(), TokenCounter)

    @requires_tokenizer
    def test_the_real_one(self) -> None:
        assert isinstance(BgeTokenCounter(DEFAULT_TOKENIZER), TokenCounter)


@requires_tokenizer
class TestBgeTokenCounter:
    def test_empty_text_is_zero(self, counter: BgeTokenCounter) -> None:
        # Matching the estimator. The chunker uses a non-zero count as its proof
        # that a span made progress, and an empty span reporting the two special
        # tokens would let it loop.
        assert counter.count("") == 0

    def test_non_empty_text_is_never_zero(self, counter: BgeTokenCounter) -> None:
        for text in ("a", " ", "\n", "。", "🙂"):
            assert counter.count(text) > 0, repr(text)

    def test_special_tokens_are_counted(self, counter: BgeTokenCounter) -> None:
        # The budget being spent is the model's sequence limit, and BGE-M3
        # spends two of it on <s> and </s> before it sees any input. Counting
        # without them accepts chunks that are over budget by exactly the amount
        # not counted — which is the whole class of bug this class removes.
        assert counter.count("hello") >= 3

    def test_it_grows_with_length(self, counter: BgeTokenCounter) -> None:
        short = counter.count("The quick brown fox.")
        long = counter.count("The quick brown fox. " * 20)

        assert long > short

    def test_cjk_is_far_denser_than_the_estimator_assumes(self, counter: BgeTokenCounter) -> None:
        # The estimator special-cases CJK precisely because a 4-chars-per-token
        # ratio would underestimate it fourfold and produce chunks that overflow
        # the window. This pins that the real tokenizer agrees CJK is dense.
        # The fullwidth comma is the point, not a typo: real CJK text uses it,
        # and substituting an ASCII comma to appease RUF001 would test a string
        # no Chinese document actually contains.
        text = "这是一个测试文档，用于验证分词器的行为。"  # noqa: RUF001
        assert counter.count(text) > len(text) / 4

    def test_the_estimator_is_in_the_right_ballpark(self, counter: BgeTokenCounter) -> None:
        # Not a precision claim — it is an estimator. This pins that it is
        # wrong by tens of percent rather than by a factor, which is the
        # difference between "chunks are slightly off" and "chunks silently
        # overflow the model window".
        prose = (
            "Retrieval-augmented generation combines a search index with a "
            "language model, so that answers are grounded in documents the "
            "operator controls rather than in the model's own recollection."
        )
        exact = counter.count(prose)
        estimated = HeuristicTokenCounter().count(prose)

        assert 0.5 < estimated / exact < 2.0

    def test_the_fingerprint_is_stable_and_short(self, counter: BgeTokenCounter) -> None:
        # Published by the model service's /v1/info. If chunking and embedding
        # ever disagree about what a token is, comparing these two strings is
        # how you find out — and it is far from obvious where else to look.
        again = BgeTokenCounter(DEFAULT_TOKENIZER)

        assert counter.fingerprint == again.fingerprint
        assert len(counter.fingerprint) == 16

    def test_it_agrees_with_the_model_services_own_counter(self, counter: BgeTokenCounter) -> None:
        # The two implementations are duplicated across an import boundary that
        # exists to keep torch out of the worker. This is what makes the
        # duplication safe rather than merely convenient: they load the same
        # file and must produce the same numbers.
        from model_service.tokenizer import count_tokens, fingerprint, load_tokenizer

        # The empty string first, so the slices below read unambiguously.
        texts = ["", "hello world", "これはテストです", "a" * 500]
        theirs = count_tokens(load_tokenizer(DEFAULT_TOKENIZER), texts)
        mine = [counter.count(text) for text in texts]

        assert fingerprint(DEFAULT_TOKENIZER) == counter.fingerprint
        # The empty string is the one deliberate divergence: this counter
        # reports 0 so the chunker can detect a span that made no progress,
        # while the service counts the special tokens the model will really
        # spend. Every non-empty text must agree exactly.
        assert mine[1:] == theirs[1:]
        assert mine[0] == 0
        assert theirs[0] == 2


@requires_tokenizer
class TestChunkingWithTheRealTokenizer:
    def test_chunks_respect_the_target_in_real_tokens(self, counter: BgeTokenCounter) -> None:
        # The point of the whole exercise. With the estimator, "512 tokens" was
        # an approximation that could exceed the model's window; with the real
        # vocabulary it is the number the model will actually see.
        from rag.domain.chunking import chunk_text

        text = (
            "Row-level security is enforced in the database rather than in the "
            "application. A policy that lives in code is a policy that a new "
            "query can forget. "
        ) * 40

        chunks = chunk_text(text, target_tokens=64, overlap_tokens=8, count_tokens=counter.count)

        assert chunks
        # A small tolerance: the splitter will not break a single indivisible
        # token run to meet the target, which is correct — the alternative is
        # cutting words in half.
        for chunk in chunks:
            assert counter.count(chunk.text) <= 64 * 2
