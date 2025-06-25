import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index
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
                byte_seq = [self.inv_vocab[bytes(b)] for b in token_bytes]
                ret.append(byte_seq)
        return ret

    def encode(self, string: str) -> list[int]:
        token_ids = self.pre_tokenization(string)

        for pair in self.params.merges.keys():
            merge_index = self.params.merges[pair]
            for ids in token_ids:
                for i in range(len(ids) - 1):
                    if (ids[i], ids[i + 1]) == pair:
                        ids[i : i + 2] = [merge_index]
                        break
        flat_indices = [index for sublist in token_ids for index in sublist]
        return flat_indices

    def decode(self, indices: list[int]) -> str:
        return ""


if __name__ == "__main__":
    pass
