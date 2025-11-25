# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
import regex
from typing import Dict, Tuple, List
from roys_bpe_tokenizer.base_tokenizer import Tokenizer
from roys_bpe_tokenizer.utils import count_adjecent, merge_ids
from collections import Counter

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str | None = None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = regex.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size > 256, "vocab_size must be greater than 256"
        num_merges = vocab_size - 256
        text_chunks = regex.findall(self.compiled_pattern, text)

        ids: List[List[int]] = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges: Dict[Tuple[int, int], int] = {}

        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            new_token_id = 256 + i
            adjecents = {}
            adjecent_counts = [count_adjecent(chunk_ids) for chunk_ids in ids]
            total_counts = Counter()
            for adjecent_count in adjecent_counts:
                total_counts += adjecent_count
            pair, count = total_counts.most_common(1)[0]
            ids = [merge_ids(chunk_ids, pair, new_token_id) for chunk_ids in ids]
            merges[pair] = new_token_id
            vocab[new_token_id] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

if __name__ == "__main__":
    tokenizer = RegexTokenizer()
    tokenizer.train("Hello, world!", 400)
    print(tokenizer.vocab)