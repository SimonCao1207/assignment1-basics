from tokenizer import BPETokenizerParams, merge, BPETokenizer
from collections import defaultdict


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
    # Start with the list of bytes of string.
    indices = list(map(int, string.encode("utf-8")))
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1

        # Find the most common pair.
        pair = max(counts, key=counts.get)
        index1, index2 = pair

        # Merge that pair.
        new_index = 256 + i
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        indices = merge(indices, pair, new_index)
    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    string = "This is a sample string for BPE training."
    params = train_bpe(string, num_merges=3)
    print("Vocabulary:", params.vocab)
    print("Merges:", params.merges)
    tokenizer = BPETokenizer(params)
    string = "the quick brown fox"
    indices = tokenizer.encode(string)
    reconstructed_string = tokenizer.decode(indices)
    print(reconstructed_string)
    assert string == reconstructed_string
