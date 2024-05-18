try:
    from transformers import AutoTokenizer
    from collections import defaultdict
    import json
except ImportError as e:
    print(f"Error importing module: {e}")
    print("Attempting to install required libraries...")
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "transformers", "collections", "json"])
        print("Libraries installed successfully.")
        from transformers import AutoTokenizer
        from collections import defaultdict
        import json
    except Exception as e:
        print(f"Error installing required libraries: {e}")

class BPE():
    """
    Byte-Pair Encoding Subword-Level Tokenization algorithm
    """
    def __init__(self):

        # pre-tokenize the corpus into words, BERT pre-tokenizer is used here
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = {}
        self.vocab = None


    def fit_on_text(self, corpus, vocab_size):
        """Train BPE tokenizer."""
        self.corpus = corpus
        self.vocab_size = vocab_size

        # compute the frequencies of each word in the corpus
        for text in self.corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] += 1

        # compute the base vocabulary of all characters in the corpus
        alphabet = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()

        # add the special token </w> at the beginning of the vocabulary
        self.vocab = ["</w>"] + alphabet.copy()

        # split each word into individual characters before training
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

        # merge the most frequent pair iteratively until the vocabulary size is reached
        while len(self.vocab) < self.vocab_size:

            # compute the frequency of each pair
            pair_freqs = self.compute_pair_freqs()

            # find the most frequent pair
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            # merge the most frequent pair
            self.splits = self.merge_pair(*best_pair)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])
        return self.merges
    def compute_pair_freqs(self):
        """Compute the frequency of each pair."""

        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def __get_vocab__(self):
      return self.vocab

    def merge_pair(self, a, b):
        """Merge the given pair."""

        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits
    def tokenize(self, text):
        """Tokenize a given text with trained BPE tokenizer (including pre-tokenization, split, and merge)."""

        pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits_text = [[l for l in word] for word in pre_tokenized_text]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits_text):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits_text[idx] = split
        return sum(splits_text, [])

    def save_vocab(self, file_path):
        """Save the vocabulary to a file."""
        # Convert the dictionary keys (tuples) to lists for JSON serialization
        merged_list = [({"pair": list(k), "merge": v}) for k, v in self.merges.items()]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_list, f)

    def load_vocab(self, file_path):
        """Load the vocabulary from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            merged_list = json.load(f)
            # Convert the list of dictionaries back to the original dictionary format
            for d in merged_list:
              self.merges[tuple(d['pair'])] = d['merge']
