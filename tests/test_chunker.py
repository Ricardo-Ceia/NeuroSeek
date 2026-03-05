"""
Tests for neuroseek/chunker.py — chunk_text()

Run with:
    PYTHONPATH=/home/ricardo/NeuroSeek ~/.local/bin/pytest tests/test_chunker.py -q
"""

import pytest
from neuroseek.chunker import chunk_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _words(text: str) -> list[str]:
    return text.split()


def make_text(n_words: int, word: str = "word") -> str:
    return " ".join(f"{word}{i}" for i in range(n_words))


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

class TestBasicBehaviour:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\t  ") == []

    def test_single_word_produces_one_chunk(self):
        result = chunk_text("hello", chunk_size=5, chunk_overlap=0)
        assert len(result) == 1
        assert result[0]["text"] == "hello"
        assert result[0]["chunk_index"] == 0

    def test_text_shorter_than_chunk_size_produces_one_chunk(self):
        text = make_text(10)
        result = chunk_text(text, chunk_size=256, chunk_overlap=32)
        assert len(result) == 1
        assert result[0]["text"] == text

    def test_text_exactly_chunk_size_produces_one_chunk(self):
        text = make_text(5)
        result = chunk_text(text, chunk_size=5, chunk_overlap=0)
        assert len(result) == 1
        assert result[0]["text"] == text

    def test_text_one_word_over_chunk_size_produces_two_chunks_no_overlap(self):
        text = make_text(6)
        result = chunk_text(text, chunk_size=5, chunk_overlap=0)
        assert len(result) == 2
        assert _words(result[0]["text"]) == _words(text)[:5]
        assert _words(result[1]["text"]) == _words(text)[5:]

    def test_chunk_indices_are_zero_based_sequential(self):
        text = make_text(20)
        result = chunk_text(text, chunk_size=5, chunk_overlap=0)
        indices = [c["chunk_index"] for c in result]
        assert indices == list(range(len(result)))

    def test_chunk_text_key_present_in_every_chunk(self):
        text = make_text(20)
        result = chunk_text(text, chunk_size=5, chunk_overlap=0)
        for chunk in result:
            assert "text" in chunk

    def test_chunk_index_key_present_in_every_chunk(self):
        text = make_text(20)
        result = chunk_text(text, chunk_size=5, chunk_overlap=0)
        for chunk in result:
            assert "chunk_index" in chunk


# ---------------------------------------------------------------------------
# Chunk count arithmetic
# ---------------------------------------------------------------------------

class TestChunkCount:
    def test_no_overlap_chunk_count(self):
        # 20 words, chunk_size=5, overlap=0 → stride=5 → 4 chunks
        text = make_text(20)
        result = chunk_text(text, chunk_size=5, chunk_overlap=0)
        assert len(result) == 4

    def test_with_overlap_chunk_count(self):
        # 10 words, chunk_size=4, overlap=2 → stride=2 → starts: 0,2,4,6,8 → 5 chunks
        text = make_text(10)
        result = chunk_text(text, chunk_size=4, chunk_overlap=2)
        assert len(result) == 5

    def test_last_chunk_smaller_than_chunk_size(self):
        # 7 words, chunk_size=5, overlap=0 → 2 chunks; last has 2 words
        text = make_text(7)
        result = chunk_text(text, chunk_size=5, chunk_overlap=0)
        assert len(result) == 2
        assert len(_words(result[1]["text"])) == 2

    def test_exact_multiple_no_leftover(self):
        text = make_text(9)
        result = chunk_text(text, chunk_size=3, chunk_overlap=0)
        assert len(result) == 3
        assert all(len(_words(c["text"])) == 3 for c in result)

    def test_single_word_chunk_size(self):
        text = make_text(5)
        result = chunk_text(text, chunk_size=1, chunk_overlap=0)
        assert len(result) == 5
        for i, chunk in enumerate(result):
            assert chunk["text"] == f"word{i}"


# ---------------------------------------------------------------------------
# Overlap correctness
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_overlap_words_are_shared_between_consecutive_chunks(self):
        # chunk_size=4, overlap=2 → last 2 words of chunk N == first 2 words of chunk N+1
        text = make_text(10)
        words = _words(text)
        result = chunk_text(text, chunk_size=4, chunk_overlap=2)
        for i in range(len(result) - 1):
            tail = _words(result[i]["text"])[-2:]
            head = _words(result[i + 1]["text"])[:2]
            assert tail == head, f"Overlap mismatch between chunk {i} and {i+1}"

    def test_overlap_one_word(self):
        text = make_text(6)
        words = _words(text)
        result = chunk_text(text, chunk_size=3, chunk_overlap=1)
        # stride = 2 → starts: 0, 2, 4
        assert len(result) == 3
        assert _words(result[0]["text"])[-1] == _words(result[1]["text"])[0]

    def test_overlap_equal_to_chunk_size_minus_one(self):
        # chunk_size=5, overlap=4 → stride=1 → starts: 0,1,2,3,4,5,6,7 → 8 chunks
        text = make_text(8)
        result = chunk_text(text, chunk_size=5, chunk_overlap=4)
        assert len(result) == 8
        for i in range(len(result) - 1):
            cur = _words(result[i]["text"])
            nxt = _words(result[i + 1]["text"])
            # The overlap between consecutive chunks is min(4, len(nxt)) words
            overlap = min(4, len(nxt))
            assert cur[1 : 1 + overlap] == nxt[:overlap]

    def test_zero_overlap_no_shared_words(self):
        text = make_text(10)
        words = _words(text)
        result = chunk_text(text, chunk_size=5, chunk_overlap=0)
        assert len(result) == 2
        assert set(_words(result[0]["text"])).isdisjoint(set(_words(result[1]["text"])))


# ---------------------------------------------------------------------------
# Word coverage
# ---------------------------------------------------------------------------

class TestWordCoverage:
    def test_no_overlap_covers_all_words_exactly_once(self):
        text = make_text(15)
        result = chunk_text(text, chunk_size=5, chunk_overlap=0)
        all_words = []
        for chunk in result:
            all_words.extend(_words(chunk["text"]))
        assert all_words == _words(text)

    def test_with_overlap_first_and_last_words_covered(self):
        text = make_text(10)
        words = _words(text)
        result = chunk_text(text, chunk_size=4, chunk_overlap=2)
        assert _words(result[0]["text"])[0] == words[0]
        assert _words(result[-1]["text"])[-1] == words[-1]

    def test_words_in_first_chunk_are_first_words_of_text(self):
        text = make_text(20)
        words = _words(text)
        result = chunk_text(text, chunk_size=5, chunk_overlap=2)
        assert _words(result[0]["text"]) == words[:5]

    def test_all_original_words_appear_in_at_least_one_chunk(self):
        text = make_text(13)
        words = set(_words(text))
        result = chunk_text(text, chunk_size=5, chunk_overlap=2)
        found = set()
        for chunk in result:
            found.update(_words(chunk["text"]))
        assert words == found


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_none_produces_no_extra_keys(self):
        result = chunk_text("hello world", chunk_size=5, chunk_overlap=0, metadata=None)
        assert set(result[0].keys()) == {"text", "chunk_index"}

    def test_metadata_fields_present_in_every_chunk(self):
        meta = {"path": "/docs/file.txt", "filetype": "txt"}
        text = make_text(20)
        result = chunk_text(text, chunk_size=5, chunk_overlap=0, metadata=meta)
        for chunk in result:
            assert chunk["path"] == "/docs/file.txt"
            assert chunk["filetype"] == "txt"

    def test_metadata_int_value(self):
        meta = {"page": 3}
        result = chunk_text("one two three", chunk_size=5, chunk_overlap=0, metadata=meta)
        assert result[0]["page"] == 3

    def test_metadata_float_value(self):
        meta = {"score": 0.95}
        result = chunk_text("a b c", chunk_size=5, chunk_overlap=0, metadata=meta)
        assert result[0]["score"] == 0.95

    def test_metadata_bool_value(self):
        meta = {"verified": True}
        result = chunk_text("x y z", chunk_size=5, chunk_overlap=0, metadata=meta)
        assert result[0]["verified"] is True

    def test_metadata_not_mutated_across_chunks(self):
        meta = {"path": "/a.txt"}
        text = make_text(10)
        result = chunk_text(text, chunk_size=3, chunk_overlap=0, metadata=meta)
        result[0]["path"] = "MODIFIED"
        assert result[1]["path"] == "/a.txt"

    def test_original_metadata_dict_not_mutated(self):
        meta = {"path": "/a.txt"}
        chunk_text("hello world", chunk_size=5, chunk_overlap=0, metadata=meta)
        assert meta == {"path": "/a.txt"}

    def test_chunk_index_not_overridden_by_metadata(self):
        # chunk_index is set first, then metadata is merged — if metadata has
        # "chunk_index" it would collide; this test documents the behaviour
        # (metadata wins, as per the docstring: "caller fields take precedence")
        meta = {"chunk_index": 99}
        result = chunk_text("a b c d e f", chunk_size=3, chunk_overlap=0, metadata=meta)
        # metadata's chunk_index=99 overrides the built-in — documented behaviour
        for chunk in result:
            assert chunk["chunk_index"] == 99

    def test_empty_metadata_dict_same_as_none(self):
        r1 = chunk_text("hello world", chunk_size=5, chunk_overlap=0, metadata=None)
        r2 = chunk_text("hello world", chunk_size=5, chunk_overlap=0, metadata={})
        assert r1 == r2


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_chunk_size_is_256(self):
        text = make_text(300)
        result = chunk_text(text)
        assert len(_words(result[0]["text"])) == 256

    def test_default_chunk_overlap_is_32(self):
        text = make_text(300)
        result = chunk_text(text)
        tail = _words(result[0]["text"])[-32:]
        head = _words(result[1]["text"])[:32]
        assert tail == head

    def test_default_produces_expected_second_chunk_start(self):
        # stride = 256 - 32 = 224
        text = make_text(300)
        words = _words(text)
        result = chunk_text(text)
        assert _words(result[1]["text"])[0] == words[224]


# ---------------------------------------------------------------------------
# Type and value errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_text_not_str_raises_type_error(self):
        with pytest.raises(TypeError, match="text must be a str"):
            chunk_text(123)  # type: ignore[arg-type]

    def test_text_none_raises_type_error(self):
        with pytest.raises(TypeError, match="text must be a str"):
            chunk_text(None)  # type: ignore[arg-type]

    def test_chunk_size_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            chunk_text("hello", chunk_size=0)

    def test_chunk_size_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            chunk_text("hello", chunk_size=-5)

    def test_chunk_overlap_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            chunk_text("hello", chunk_size=5, chunk_overlap=-1)

    def test_chunk_overlap_equals_chunk_size_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
            chunk_text("hello", chunk_size=5, chunk_overlap=5)

    def test_chunk_overlap_greater_than_chunk_size_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
            chunk_text("hello", chunk_size=5, chunk_overlap=10)

    def test_chunk_size_bool_raises_type_error(self):
        with pytest.raises(TypeError, match="chunk_size must be an int"):
            chunk_text("hello", chunk_size=True)  # type: ignore[arg-type]

    def test_chunk_overlap_bool_raises_type_error(self):
        with pytest.raises(TypeError, match="chunk_overlap must be an int"):
            chunk_text("hello", chunk_size=5, chunk_overlap=False)  # type: ignore[arg-type]

    def test_chunk_size_float_raises_type_error(self):
        with pytest.raises(TypeError, match="chunk_size must be an int"):
            chunk_text("hello", chunk_size=5.0)  # type: ignore[arg-type]

    def test_metadata_non_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="metadata must be a dict or None"):
            chunk_text("hello", metadata="bad")  # type: ignore[arg-type]

    def test_metadata_non_str_key_raises_type_error(self):
        with pytest.raises(TypeError, match="metadata keys must be str"):
            chunk_text("hello", metadata={1: "v"})  # type: ignore[arg-type]

    def test_metadata_invalid_value_type_raises_type_error(self):
        with pytest.raises(TypeError, match="metadata values must be"):
            chunk_text("hello", metadata={"key": [1, 2, 3]})  # type: ignore[arg-type]

    def test_metadata_nested_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="metadata values must be"):
            chunk_text("hello", metadata={"key": {"nested": "dict"}})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_chunk_size_1_overlap_0_each_word_is_own_chunk(self):
        text = "alpha beta gamma"
        result = chunk_text(text, chunk_size=1, chunk_overlap=0)
        assert [c["text"] for c in result] == ["alpha", "beta", "gamma"]

    def test_multiple_spaces_treated_as_one_separator(self):
        text = "hello   world   foo"
        result = chunk_text(text, chunk_size=10, chunk_overlap=0)
        assert result[0]["text"] == "hello world foo"

    def test_leading_trailing_whitespace_ignored(self):
        text = "  hello world  "
        result = chunk_text(text, chunk_size=10, chunk_overlap=0)
        assert result[0]["text"] == "hello world"

    def test_newlines_treated_as_whitespace(self):
        text = "line one\nline two\nline three"
        result = chunk_text(text, chunk_size=10, chunk_overlap=0)
        assert len(result) == 1
        assert "line" in result[0]["text"]

    def test_tabs_treated_as_whitespace(self):
        text = "col1\tcol2\tcol3"
        result = chunk_text(text, chunk_size=5, chunk_overlap=0)
        assert result[0]["text"] == "col1 col2 col3"

    def test_single_chunk_size_equals_text_length(self):
        text = make_text(5)
        result = chunk_text(text, chunk_size=5, chunk_overlap=0)
        assert result[0]["text"] == text

    def test_large_overlap_many_chunks(self):
        text = make_text(10)
        # chunk_size=5, overlap=4 → stride=1 → starts: 0..9 → 10 chunks
        result = chunk_text(text, chunk_size=5, chunk_overlap=4)
        assert len(result) == 10
        assert all("text" in c and "chunk_index" in c for c in result)
