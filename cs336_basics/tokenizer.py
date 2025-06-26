import json
import os
import pathlib
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
        self.inv_vocab: dict[bytes, int] = {v: k for k, v in params.vocab.items()}  # bytes -> index

    def pre_tokenization(self, chunk: str) -> list[list[int]]:
        """
        Perform regex-based pre-tokenization and return a list of pre-token as a sequence of byte indices.
        """
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

        pattern = "|".join(re.escape(tok) for tok in sorted_special_tokens)
        split_on_special_tokens = re.split(f"({pattern})", chunk)
        ret = []
        if DEBUG:
            print(f"Split on special tokens: {split_on_special_tokens}")
        for span in split_on_special_tokens:
            if span in self.special_tokens:
                ret.append([self.inv_vocab[span.encode("utf-8")]])
                continue

            for match in re.finditer(PAT, span):
                token = match.group(0)
                token_bytes = token.encode("utf-8")
                byte_seq = [self.inv_vocab[bytes([b])] for b in token_bytes]
                ret.append(byte_seq)
        return ret

    def encode(self, string: str) -> list[int]:
        token_ids = self.pre_tokenization(string)
        if DEBUG:
            print(f"Pre-tokenized: {token_ids}")

        for pair in self.params.merges:
            merge_index = self.inv_vocab[b"".join(pair)]
            for ids in token_ids:
                i = 0
                while i < len(ids) - 1:
                    if (self.params.vocab[ids[i]], self.params.vocab[ids[i + 1]]) == pair:
                        ids[i : i + 2] = [merge_index]
                    else:
                        i += 1
        flat_indices = [index for sublist in token_ids for index in sublist]
        return flat_indices

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)

    def decode(self, indices: list[int]) -> str:
        return b"".join([self.params.vocab[idx] for idx in indices]).decode("utf-8", errors="replace")


if __name__ == "__main__":
    # special_tokens = ["<|endoftext|>"]
    # vocabs = {x: bytes([x]) for x in range(256)}
    # vocabs[256] = special_tokens[0].encode("utf-8")
    # vocabs.update({257: b"st", 258: b"est", 259: b"ow", 260: b"low", 261: b"west", 262: b"ne"})
    # merges = [(b"s", b"t"), (b"e", b"st"), (b"o", b"w"), (b"l", b"ow"), (b"w", b"est"), (b"n", b"e")]
    # params = BPETokenizerParams(vocab=vocabs, merges=merges, special_tokens=special_tokens)
    # tokenizer = BPETokenizer(params)
    # test_string = "This is a test string to tokenize. low and west are examples."
    # encoded = tokenizer.encode(test_string)
    # print(f"Encoded: {encoded}")
    # decoded = tokenizer.decode(encoded)
    # assert decoded == test_string, f"Decoded string does not match original: {decoded} != {test_string}"

    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        ids = []
        for _id in tokenizer.encode_iterable(f):
            ids.append(_id)
