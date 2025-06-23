from collections import defaultdict
import pathlib
import os
from typing import BinaryIO

import regex as re
from cs336_basics.tokenizer import BPETokenizer, BPETokenizerParams
import multiprocessing as mp

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


def get_pair_counts(freqs: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    pair_counts = defaultdict(int)
    for byte_seq, count in freqs.items():
        for i in range(len(byte_seq) - 1):
            pair = (byte_seq[i], byte_seq[i + 1])
            pair_counts[pair] += count
    return pair_counts


def initial_pretoken_freqs(chunk: str) -> dict[tuple[int, ...], int]:
    """
    Perform regex-based pre-tokenization and return a frequency dictionary
    of each pre-token as a sequence of byte indices.
    """
    freqs = defaultdict(int)
    for match in re.finditer(PAT, chunk):
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


def merge(
    byte_seq_freq: dict[tuple[int, ...], int], pair: tuple[int, int], new_index: int
) -> dict[tuple[int, ...], int]:
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


def train_bpe(
    input_path: str | os.PathLike, num_merges: int, special_tokens: list[str] = ["<|endoftext|>"]
) -> BPETokenizerParams:
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    next_index = 256

    for tok in special_tokens:
        tok = tok.encode("utf-8")
        vocab[next_index] = tok
        next_index += 1

    num_processes = mp.cpu_count()

    chunks = read_file_chunks(input_path, num_chunks=num_processes, special_token=special_tokens[0])

    # Remove special tokens from chunks before tokenizing
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    cleaned_chunks = [re.split(f"({pattern})", chunk) for chunk in chunks]
    clean_spans = [part for group in cleaned_chunks for part in group if part and part not in special_tokens]

    with mp.Pool(num_processes) as pool:
        all_freqs = pool.map(initial_pretoken_freqs, clean_spans)

    byte_seq_freq = merge_dicts(all_freqs)

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

        if DEBUG:
            print(f"Merge {i + 1}: {vocab[pair[0]]} + {vocab[pair[1]]} -> {vocab[next_index]}")
            print(byte_seq_freq, "\n")

        next_index += 1

    return BPETokenizerParams(vocab=vocab, merges=merges, special_tokens=special_tokens)


if __name__ == "__main__":
    # input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    input_path = os.path.join("data", "tiny.txt")
    special_tokens = ["<|endoftext|>"]
    params = train_bpe(input_path, num_merges=12, special_tokens=special_tokens)
    # print("Vocabulary:", params.vocab)
    print("Merges:", params.merges)
    tokenizer = BPETokenizer(params)
    string = "the quick brown fox"
    indices = tokenizer.encode(string)
    reconstructed_string = tokenizer.decode(indices)
    assert string == reconstructed_string
