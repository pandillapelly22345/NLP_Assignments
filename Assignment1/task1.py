import re
import json
from collections import defaultdict
import time

class WordPieceTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocabulary = {}   # vocabulary of subwords
        self.split_corpus = []  # split corpus where each word is broken into subwords

    def preprocess_data(self, text):
        text = text.lower()   # Convert text to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        tokens = [token for token in text.split() if token]  # Split into tokens and remove empty strings
        return tokens

    def initialize_vocabulary(self, tokens):
        vocabulary = set()
        for token in tokens:
            if not token:
                continue
            chars = list(token) # Convert token into a list of characters
            prefixed_chars = [chars[0]]  # Start with the first character
            for c in chars[1:]:  # Loop through the rest of the characters
                prefixed_chars.append(f'##{c}')  # Add the '##' prefix to each character
            vocabulary.update(prefixed_chars)
        self.vocabulary = {char: idx for idx, char in enumerate(sorted(vocabulary))}
        self.vocabulary = self.adding_special_tokens(self.vocabulary)

    def build_split_corpus(self, tokens):
        split_corpus = []
        for token in tokens:
            if not token:
                continue
            chars = list(token)  # Convert token into a list of characters
            prefixed_chars = [chars[0]]  # Start with the first character
            for c in chars[1:]:  # Loop through the rest of the characters
                prefixed_chars.append(f'##{c}')  # Add the '##' prefix to each character
            split_corpus.append(prefixed_chars)
        self.split_corpus = split_corpus

    def count_pairs(self):
        pair_counts = defaultdict(int)
        for word in self.split_corpus:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])  # Create adjacent subword pairs
                pair_counts[pair] += 1   # Increment count for the pair
        return pair_counts

    def compute_token_frequencies(self):
        token_freq = defaultdict(int)
        for word in self.split_corpus:
            for token in word:
                token_freq[token] += 1  # Increment frequency for each token
        return token_freq

    def calculate_pair_scores(self, pair_counts):
        token_freq = self.compute_token_frequencies()  # Get token frequencies
        scores = {}
        for pair, freq in pair_counts.items():
            first_freq = token_freq[pair[0]]  # Frequency of first token in pair
            second_freq = token_freq[pair[1]] # Frequency of second token in pair
            if first_freq * second_freq != 0:
                score = freq / (first_freq * second_freq)  # Calculate pair score
            else:
                score = 0
            scores[pair] = score   # Store the score
        return scores

    def merge_pair(self, a, b):  # Merge subwords a and b
        if b.startswith("##"):
            merge = a + b[2:]
        else:
            merge = a + b
        new_splits = []
        for word in self.split_corpus:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:  # If pair found, merge them
                    new_word.append(merge)
                    i += 2  # Skip the next element as it's already merged
                else:
                    new_word.append(word[i])  # Keep the token as is
                    i += 1
            new_splits.append(new_word)  # Add the updated word to the new corpus
        self.split_corpus = new_splits
        if merge not in self.vocabulary:    # Add new merge to the vocabulary if it's not already present
            self.vocabulary[merge] = len(self.vocabulary)

    def construct_vocabulary(self):
        while len(self.vocabulary) < self.vocab_size:
            pair_counts = self.count_pairs() # Count adjacent subword pairs
            if not pair_counts:
                break   # Exit if no more pairs are found
            scores = self.calculate_pair_scores(pair_counts)  # Calculate scores for pairs
            best_pair = max(scores, key=scores.get)   # Get the pair with the highest score
            if scores[best_pair] == 0:
                break   # Exit if no pair has a non-zero score
            self.merge_pair(*best_pair)   # Merge the best pair

    def adding_special_tokens(self, vocabulary):
        updated_vocabulary = {'[UNK]': 0, '[PAD]': 1}   # Add special tokens with assigned indices
        updated_vocabulary.update({token: idx + 2 for idx, token in enumerate(vocabulary)})  # Add existing tokens with new indices
        return updated_vocabulary

    def save_vocabulary(self, group_no):
        with open(f"vocabulary_{group_no}.txt", "w", encoding='utf-8') as file:
            for token in sorted(self.vocabulary, key=self.vocabulary.get):   # Sort tokens by index
                file.write(f"{token}\n")    # Write each token to the file

    def tokenize(self, sentence):
        tokens = []
        for word in sentence.split():   # Iterate through each word in the sentence
            current_word = word
            subword_tokens = []

            while current_word:
                matched = False
                for i in range(len(current_word), 0, -1):  # Try subwords of decreasing length
                    subword = current_word[:i]  # Get the subword
                    if subword in self.vocabulary:   # Check if subword is in the vocabulary
                        if not subword_tokens:
                            subword_tokens.append(subword)
                        else:
                            subword_tokens.append(f'##{subword}')
                        current_word = current_word[i:]    # Update the current word to remaining part
                        matched = True
                        break
                if not matched:
                    subword_tokens.append("[UNK]")  # If no match found, use [UNK]
                    break

            tokens.extend(subword_tokens)  # Add subword tokens to the list of tokens

        return tokens

    def process_test_file(self, test_file_path, group_no):
        with open(test_file_path, "r", encoding='utf-8') as file:
            data = json.load(file)   # Load the test data from JSON file

        tokenized_output = {}
        for entry in data:     # Iterate over all entries in the test data
            sentence_id = entry["id"]
            sentence = entry["sentence"]
            tokenized_output[sentence_id] = self.tokenize(sentence)  # Tokenize the sentence and store the result

        with open(f"tokenized_{group_no}.json", "w", encoding='utf-8') as file:
            json.dump(tokenized_output, file, indent=4)     # Save the tokenized output to a new JSON file

    def train(self, corpus):
        """Train the tokenizer on the given corpus."""
        if isinstance(corpus, list):  # If corpus is a list of strings, join them into a single string
            corpus = " ".join(corpus)
        tokens = self.preprocess_data(corpus)  # Preprocess the corpus
        self.initialize_vocabulary(tokens)  # Initialize vocabulary
        self.build_split_corpus(tokens)  # Build split corpus
        self.construct_vocabulary()  # Construct vocabulary

def main():
    start_time = time.time()
    print("Starting the algorithm...")

    # Configuration
    vocab_size = 1000
    corpus_path = 'corpus.txt'
    test_json_path = 'sample_test.json'
    group_no = 20
    
    # Initialize the tokenizer
    tokenizer = WordPieceTokenizer(vocab_size=vocab_size)
    
    # Load and preprocess corpus
    with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
        corpus = corpus_file.read()   # Read the entire corpus into a string
    
    # Train the tokenizer
    train_start = time.time()
    tokenizer.train(corpus)
    print(f"Training completed in {time.time() - train_start:.2f} seconds")
    
    # Save vocabulary
    tokenizer.save_vocabulary(group_no)
    print(f"Vocabulary size: {len(tokenizer.vocabulary)}")
    
    # Tokenize test file
    test_start = time.time()
    tokenizer.process_test_file(test_json_path, group_no)
    print(f"Test file processing completed in {time.time() - test_start:.2f} seconds")
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    print("Running the algorithm...")
    main()