from tokenizer import BPETokenizerParams, BPETokenizer
from collections import defaultdict
import regex as re

# Pre-tokenization pattern (used by GPT-2)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_pair_counts(freqs: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    pair_counts = defaultdict(int)
    for byte_seq, count in freqs.items():
        for i in range(len(byte_seq) - 1):
            pair = (byte_seq[i], byte_seq[i + 1])
            pair_counts[pair] += count
    return pair_counts


def initial_pretoken_freqs(text: str) -> dict[tuple[int, ...], int]:
    """
    Perform regex-based pre-tokenization and return a frequency dictionary
    of each pre-token as a sequence of byte indices.
    """
    freqs = defaultdict(int)
    for match in re.finditer(PAT, text):
        token = match.group(0)
        token_bytes = token.encode("utf-8")
        byte_seq = tuple(b for b in token_bytes)
        freqs[byte_seq] += 1
    return freqs


def merge(byte_seq_freq: dict[tuple[int, ...], int], pair: tuple[int, int], new_index: int) -> dict[tuple[int, ...], int]:
    """
    Every occurrence of most frequent byte pair in each byte sequence, for example, (“A”, “B”) is merged, i.e.,
    replaced with a new token “AB”
    """
    new_freqs = defaultdict(int)
    for byte_seq, count in byte_seq_freq.items():
        new_byte_seq = []
        i = 0
        while i < len(byte_seq):
            if i + 1 < len(byte_seq) and (byte_seq[i], byte_seq[i + 1]) == pair:
                new_byte_seq.append(new_index)
                i += 2
            else:
                new_byte_seq.append(byte_seq[i])
                i += 1
        new_freqs[tuple(new_byte_seq)] += count
    return new_freqs


def train_bpe(text: str, num_merges: int) -> BPETokenizerParams:
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    next_index = 256  # Start merging from index 256

    byte_seq_freq = initial_pretoken_freqs(text)
    for i in range(num_merges):
        pair_counts = get_pair_counts(byte_seq_freq)
        if not pair_counts:
            break

        # Find the most common pair.
        pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

        # Merge that pair.
        merges[pair] = next_index
        vocab[next_index] = vocab[pair[0]] + vocab[pair[1]]
        byte_seq_freq = merge(byte_seq_freq, pair, next_index)
        print(f"Merge {i + 1}: {vocab[pair[0]]} + {vocab[pair[1]]} -> {vocab[next_index]}")
        print(byte_seq_freq, "\n")

        next_index += 1

    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    string = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""
    params = train_bpe(string, num_merges=6)
    # print("Vocabulary:", params.vocab)
    print("Merges:", params.merges)
    tokenizer = BPETokenizer(params)
    string = "the quick brown fox"
    indices = tokenizer.encode(string)
    reconstructed_string = tokenizer.decode(indices)
    print(reconstructed_string)
    assert string == reconstructed_string
