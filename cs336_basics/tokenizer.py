import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import regex as re

DEBUG = os.environ.get("DEBUG") == "1"

# Pre-tokenization pattern (used by GPT-2)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, indices: list[int]) -> str:
        pass


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

        pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
        split_on_special_tokens = re.split(f"({pattern})", chunk)
        ret = []
        for span in split_on_special_tokens:
            for match in re.finditer(PAT, span):
                token = match.group(0)
                token_bytes = token.encode("utf-8")
                if token_bytes in self.special_tokens:
                    ret.append([self.inv_vocab[token_bytes]])
                byte_seq = [b for b in token_bytes]
                ret.append(byte_seq)
        return ret

    def encode(self, string: str) -> list[int]:
        token_ids = self.pre_tokenization(string)

        for pair in self.params.merges:
            merge_index = self.inv_vocab[b"".join(pair)]
            for ids in token_ids:
                i = 0
                while i < len(ids) - 1:
                    if (ids[i], ids[i + 1]) == pair:
                        ids[i : i + 2] = [merge_index]
                    else:
                        i += 1
        flat_indices = [index for sublist in token_ids for index in sublist]
        return flat_indices

    def decode(self, indices: list[int]) -> str:
        return b"".join([self.params.vocab[idx] for idx in indices]).decode("utf-8", errors="replace")


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocabs = {x: bytes([x]) for x in range(256)}
    vocabs[256] = special_tokens[0].encode("utf-8")
    vocabs.update({257: b"st", 258: b"est", 259: b"ow", 260: b"low", 261: b"west", 262: b"ne"})
    merges = [(b"s", b"t"), (b"e", b"st"), (b"o", b"w"), (b"l", b"ow"), (b"w", b"est"), (b"n", b"e")]
    params = BPETokenizerParams(vocab=vocabs, merges=merges, special_tokens=special_tokens)
    tokenizer = BPETokenizer(params)
    test_string = "This is a test string to tokenize. low and west are examples."
    encoded = tokenizer.encode(test_string)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    assert decoded == test_string, f"Decoded string does not match original: {decoded} != {test_string}"
