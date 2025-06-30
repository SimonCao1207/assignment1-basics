import cProfile
import json
import os
import pathlib
import pstats
import tracemalloc
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache

import regex as re

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests" / "fixtures"

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"
DEBUG = os.environ.get("DEBUG") == "1"

# Pre-tokenization pattern (used by GPT-2)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, indices: list[int]) -> str:
        pass

    @abstractmethod
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.
        This is required for memory-eﬀicient tokenization of large files
        that we cannot directly load into memory
        """
        pass


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    return BPETokenizer(
        BPETokenizerParams(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens if special_tokens is not None else ["<|endoftext|>"],
        )
    )


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""

    vocab: dict[int, bytes]  # index -> bytes
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""

    def __init__(self, params: BPETokenizerParams):
        self.params = params
        self.special_tokens = params.special_tokens if params.special_tokens is not None else ["<|endoftext|>"]
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)  # longest first
        self.vocab: dict[int, bytes] = params.vocab  # index -> bytes
        self.inv_vocab: dict[bytes, int] = {v: k for k, v in params.vocab.items()}  # bytes -> index
        self.merge_lookup: dict[tuple[int, int], tuple[int, int]] = {}
        for i, pair in enumerate(self.params.merges):
            merge_index = self.inv_vocab[b"".join(pair)]
            pair_indices = (self.inv_vocab[pair[0]], self.inv_vocab[pair[1]])
            self.merge_lookup[pair_indices] = (merge_index, i)
        self._byte_to_token: list[int] = [self.inv_vocab[bytes([b])] for b in range(256)]

    def pre_tokenization(self, chunk: str) -> list[list[int]]:
        """
        Perform regex-based pre-tokenization and return a list of pre-token as a sequence of byte indices.
        """

        pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
        split_on_special_tokens = re.split(f"({pattern})", chunk)
        ret = []
        for span in split_on_special_tokens:
            if not span:
                continue
            if span in self.special_tokens:
                ret.append([self.inv_vocab[span.encode("utf-8")]])
                continue

            for match in re.finditer(PAT, span):
                token_bytes = match.group(0).encode("utf-8")
                ret.append([self._byte_to_token[b] for b in token_bytes])
        return ret

    def encode(self, string: str) -> list[int]:
        # Process in smaller chunks and yield results to avoid holding large lists
        if len(string) > 10000:
            return list(self._encode_generator(string, chunk_size=512))
        else:
            return self._encode_chunk(string)

    def _encode_generator(self, string: str, chunk_size: int) -> Iterator[int]:
        for i in range(0, len(string), chunk_size):
            chunk = string[i : i + chunk_size]
            yield from self._encode_chunk(chunk)

    def _encode_chunk(self, chunk: str) -> list[int]:
        token_ids = self.pre_tokenization(chunk)
        result = []
        for ids in token_ids:
            if len(ids) == 1:
                result.extend(ids)
                continue
            while True:
                best_priority = float("inf")
                best_pos = -1
                best_merge_index = None
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    if pair in self.merge_lookup:
                        merge_index, priority = self.merge_lookup[pair]
                        if priority < best_priority:
                            best_priority = priority
                            best_pos = i
                            best_merge_index = merge_index
                if not best_merge_index:
                    break
                ids[best_pos] = best_merge_index
                ids.pop(best_pos + 1)
            result.extend(ids)
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)

    def decode(self, indices: list[int]) -> str:
        return b"".join([self.vocab[idx] for idx in indices]).decode("utf-8", errors="replace")


def display_top_memory_consumers(snapshot, key_type="lineno", limit=10):
    """Display top memory consumers from tracemalloc snapshot."""
    top_stats = snapshot.statistics(key_type)

    print(f"\nTop {limit} memory consumers by {key_type}:")
    print("-" * 80)

    for index, stat in enumerate(top_stats[:limit], 1):
        if "tokenzier.py" in stat.traceback.format()[0]:
            print(f"#{index}: {stat}")
            if key_type == "lineno":
                frame = stat.traceback.format()[0]
                print(f"    {frame}")
            print()


def profile_memory_detailed(tokenizer, text):
    """Memory profiling focused on pre-tokenization stage."""
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    print("=== Memory Profiling Started ===")
    print("\n1. Pre-tokenization stage...")

    pre_tokens = tokenizer.pre_tokenization(text)
    snapshot2 = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n=== Memory Usage Summary ===")
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    print("\n=== Memory Growth by Stage ===")

    pre_token_diff = snapshot2.compare_to(snapshot1, "lineno")
    total_pre_token = sum(stat.size_diff for stat in pre_token_diff)

    print(f"Pre-tokenization stage: +{total_pre_token / 1024 / 1024:.2f} MB")

    display_top_memory_consumers(snapshot2, "lineno", 15)
    display_top_memory_consumers(snapshot2, "filename", 10)

    print("\n=== Detailed Memory Growth (Pre-tokenization) ===")
    for stat in pre_token_diff[:10]:
        if stat.size_diff > 0:
            print(f"+{stat.size_diff / 1024:.1f} KB: {stat}")
    return pre_tokens


if __name__ == "__main__":
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        corpus_contents = f.read()
    profiler = cProfile.Profile()
    profiler.enable()
    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents
    profiler.disable()
    if DEBUG:
        profile_memory_detailed(tokenizer, corpus_contents)
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        this_file = pathlib.Path(__file__).name
        stats.print_stats(this_file)
