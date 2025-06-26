import cProfile
import functools
import multiprocessing as mp
import os
import pathlib
import pstats
from collections import defaultdict
from typing import BinaryIO

import regex as re

from cs336_basics.tokenizer import BPETokenizerParams

DEBUG = os.environ.get("DEBUG") == "1"

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests" / "fixtures"

# Pre-tokenization pattern (used by GPT-2)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def read_file_chunks(file_path: str | os.PathLike, num_chunks: int, special_token: str) -> list[str]:
    """
    Read a text file and split it into document-aligned chunks at special_token positions.
    """
    special_bytes = special_token.encode("utf-8")
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, special_bytes)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        return chunks


def pre_tokenization(chunk: str, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
    """
    Perform regex-based pre-tokenization and return a frequency dictionary
    of each pre-token as a sequence of byte indices.
    """

    # Remove special tokens from chunks before tokenizing
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    cleaned_chunk = re.split(f"({pattern})", chunk)
    cleaned_spans = [part for part in cleaned_chunk if part and part not in special_tokens]

    freqs = defaultdict(int)
    for span in cleaned_spans:
        for match in re.finditer(PAT, span):
            token = match.group(0)
            token_bytes = token.encode("utf-8")
            byte_seq = tuple(b for b in token_bytes)
            freqs[byte_seq] += 1
    return freqs


def merge_dicts(dicts: list[dict[tuple[int, ...], int]]) -> dict[tuple[int, ...], int]:
    """
    Merge a list of frequency dictionaries into a single dictionary.
    """
    merged_freqs = defaultdict(int)
    for d in dicts:
        for byte_seq, count in d.items():
            merged_freqs[byte_seq] += count
    return dict(merged_freqs)


class BPETrainer:
    """
    BPE trainer that maintains incremental data structures
    """

    def __init__(self, byte_seq_freq: dict[tuple[int, ...], int]):
        self.vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
        self.merges: list[tuple[bytes, bytes]] = []
        self.next_index = 256
        self.byte_seq_freq = byte_seq_freq  # byte sequence -> frequency
        self.pair_counts: dict[tuple[int, int], int] = defaultdict(int)  # (byte1, byte2) -> frequency
        self.pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = {}  # (byte1, byte2) -> (word1, word2, ...)

    def update_pair_structures(self):
        """
        Update pair counts and pair to words structures based on current byte sequence frequencies.
        """
        self.pair_counts.clear()
        self.pair_to_words.clear()

        for byte_seq, count in self.byte_seq_freq.items():
            for i in range(len(byte_seq) - 1):
                pair = (byte_seq[i], byte_seq[i + 1])
                self.pair_counts[pair] += count
                if pair not in self.pair_to_words:
                    self.pair_to_words[pair] = set()
                self.pair_to_words[pair].add(byte_seq)

    def get_most_frequent_pair(self) -> tuple[int, int] | None:
        """
        Find the most frequent byte pair in the current pair counts.
        Returns None if no pairs are found.
        """
        if not self.pair_counts:
            return None
        max_freq = max(self.pair_counts.values())
        tied_pairs = [pair for pair, freq in self.pair_counts.items() if freq == max_freq]

        if len(tied_pairs) > 1:
            return max(tied_pairs, key=lambda p: (self.vocab[p[0]], self.vocab[p[1]]))
        return tied_pairs[0]

    def _merge_word(self, old_word: tuple[int, ...], pair: tuple[int, int]) -> tuple[int, ...]:
        """
        Merge a byte pair in a word and return the new word.
        """
        new_word = []
        i = 0
        while i < len(old_word):
            if i + 1 < len(old_word) and (old_word[i], old_word[i + 1]) == pair:
                new_word.append(self.next_index)
                i += 2
            else:
                new_word.append(old_word[i])
                i += 1
        return tuple(new_word)

    def _remove_word_contribution(self, old_word: tuple[int, ...], freq: int) -> None:
        """
        Remove the contribution of a old word from the pair_counts and pair_to_words structures.
        """
        for i in range(len(old_word) - 1):
            pair = (old_word[i], old_word[i + 1])
            self.pair_counts[pair] -= freq
            if self.pair_counts[pair] <= 0:
                del self.pair_counts[pair]
            if pair in self.pair_to_words:
                self.pair_to_words[pair].discard(old_word)
                if not self.pair_to_words[pair]:
                    del self.pair_to_words[pair]

    def _add_word_contribution(self, new_word: tuple[int, ...], freq: int) -> None:
        """
        Add the contribution of a new word from the pair_counts and pair_to_words structures.
        """
        for i in range(len(new_word) - 1):
            pair = (new_word[i], new_word[i + 1])
            self.pair_counts[pair] += freq
            if pair not in self.pair_to_words:
                self.pair_to_words[pair] = set()
            self.pair_to_words[pair].add(new_word)

    def merge_pair(self, old_pair: tuple[int, int]) -> None:
        """
        Merge the most frequent byte pair in the byte sequence frequencies.
        """
        old_pair_bytes = (self.vocab[old_pair[0]], self.vocab[old_pair[1]])
        self.merges.append(old_pair_bytes)
        self.vocab[self.next_index] = self.vocab[old_pair[0]] + self.vocab[old_pair[1]]
        affected_words = self.pair_to_words[old_pair]
        del self.pair_to_words[old_pair]
        del self.pair_counts[old_pair]
        for old_word in affected_words:
            freq = self.byte_seq_freq[old_word]
            new_word = self._merge_word(old_word, old_pair)
            self._remove_word_contribution(old_word, freq)

            del self.byte_seq_freq[old_word]
            self.byte_seq_freq[new_word] = self.byte_seq_freq.get(new_word, 0) + freq

            self._add_word_contribution(new_word, self.byte_seq_freq[new_word])
        self.next_index += 1

    def add_special_token(self, special_tokens: list[str]):
        """
        Add a special token to the vocabulary and return its index.
        """
        for tok in special_tokens:
            tok = tok.encode("utf-8")
            self.vocab[self.next_index] = tok
            self.next_index += 1


def train_bpe(
    input_path: str | os.PathLike, num_merges: int, special_tokens: list[str] = ["<|endoftext|>"]
) -> BPETokenizerParams:
    num_processes = mp.cpu_count()

    chunks = read_file_chunks(input_path, num_chunks=num_processes, special_token=special_tokens[0])

    with mp.Pool(num_processes) as pool:
        func = functools.partial(pre_tokenization, special_tokens=special_tokens)
        all_freqs = pool.map(func, chunks)

    byte_seq_freq = merge_dicts(all_freqs)
    trainer = BPETrainer(byte_seq_freq)
    trainer.update_pair_structures()
    trainer.add_special_token(special_tokens)

    for _ in range(num_merges):
        pair = trainer.get_most_frequent_pair()
        if pair is None:
            break
        trainer.merge_pair(pair)

    if DEBUG:
        merge_seq = [(pair[0].decode(errors="ignore"), pair[1].decode(errors="ignore")) for pair in trainer.merges]
        print("Num processes:", num_processes)
        print("Sequence of merges:", merge_seq)

    return BPETokenizerParams(vocab=trainer.vocab, merges=trainer.merges, special_tokens=special_tokens)


if __name__ == "__main__":
    dataset = "TinyStoriesV2-GPT4-train"
    input_path = os.path.join("/data/namcao/cs336", f"{dataset}.txt")
    # input_path = os.path.join("data", "tiny.txt")
    special_tokens = ["<|endoftext|>"]
    with cProfile.Profile() as pr:
        params = train_bpe(input_path, num_merges=10000, special_tokens=special_tokens)

    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats("cumulative")

    this_file = pathlib.Path(__file__).name
    stats.print_stats(this_file)
