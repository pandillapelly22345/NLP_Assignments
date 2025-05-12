import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from task1 import WordPieceTokenizer  # Import WordPieceTokenizer from Task 1

# Load the corpus from a file (sentence-by-sentence)
def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]  # Read as list of sentences
    return corpus

# Load corpus as list of sentences
corpus = load_corpus("corpus.txt")

# Initialize WordPieceTokenizer
tokenizer = WordPieceTokenizer(vocab_size=1000)
tokenizer.train(" ".join(corpus))  # Train tokenizer on entire text corpus

# Tokenize **sentence by sentence**, but keep only full words
tokenized_sentences = []
for sentence in corpus:
    tokenized_sentences.append(tokenizer.tokenize(sentence))

# Flatten into one long list of tokens, **excluding subword fragments**
tokens = []
for sentence in tokenized_sentences:
    for token in sentence:
        if not token.startswith("##"):
            tokens.append(token)

# Build vocabulary
vocab = Counter(tokens)
word_to_idx = {word: idx + 2 for idx, (word, _) in enumerate(vocab.items())}    # Assign indices starting from 2
word_to_idx['[PAD]'] = 0    # Padding token
word_to_idx['[UNK]'] = 1    # Unknown token
idx_to_word = {idx: word for word, idx in word_to_idx.items()}   # Reverse lookup
vocab_size = len(word_to_idx)

# Debugging: Print some tokenized sentences
print("Example Tokenized Sentences:")
for i in range(1):
    print(tokenized_sentences[i])

# One-hot encoding function
def one_hot_encode(index, vocab_size):
    one_hot = np.zeros(vocab_size)
    one_hot[index] = 1
    return one_hot

# Word2Vec dataset class
class Word2VecDataset(Dataset):
    def __init__(self, tokens, window_size=4):  # window size - 4
        self.data = []
        for i, word in enumerate(tokens):  # Generate context for each word within the specified window size
            context_indices = [
                word_to_idx.get(tokens[j], word_to_idx['[UNK]'])     # Get indices of context words
                for j in range(i - window_size, i + window_size + 1)
                if j != i and 0 <= j < len(tokens)
            ]

            if context_indices:    # Create one-hot vectors for context words and calculate the mean
                context_vectors = np.array([one_hot_encode(idx, vocab_size) for idx in context_indices])
                context_mean = np.mean(context_vectors, axis=0)    # Average context vectors
                target = word_to_idx.get(word, word_to_idx['[UNK]'])   # Get index for the target word
                self.data.append((context_mean, target))    # Store the pair

    def __len__(self):
        return len(self.data)   # Return the number of samples

    def __getitem__(self, idx):
        context_vector, target = self.data[idx]    # Return context and target pair
        return torch.tensor(context_vector, dtype=torch.float32), torch.tensor(target)

# Word2Vec model with dropout and batch normalization
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2VecModel, self).__init__()
        self.fc1 = nn.Linear(vocab_size, embed_size)   # Fully connected layer for embedding
        self.bn1 = nn.BatchNorm1d(embed_size)    # Batch normalization to stabilize training
        self.dropout1 = nn.Dropout(0.3)    # Dropout layer for regularization
        self.fc2 = nn.Linear(embed_size, vocab_size)   # Output layer to predict the target word

    def forward(self, context):    # Forward pass: Apply ReLU, batch normalization, dropout, and then output layer
        hidden = self.dropout1(self.bn1(torch.relu(self.fc1(context))))
        return self.fc2(hidden)

# Training function with improved learning

def train(tokens, vocab_size, embed_size=50, window_size=4, epochs=10, lr=0.001, batch_size=64):
    dataset = Word2VecDataset(tokens, window_size)   # Prepare dataset
    train_size = int(0.8 * len(dataset))   # 80% of data for training
    val_size = len(dataset) - train_size   # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # DataLoader for training
    val_loader = DataLoader(val_dataset, batch_size=batch_size)   # DataLoader for validation
 
    model = Word2VecModel(vocab_size, embed_size)   # Initialize the model
    loss_fn = nn.CrossEntropyLoss()   # Loss function for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr)    # Adam optimizer for training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)   # Learning rate scheduler

    best_val_loss = float('inf')    # Track the best validation loss
    patience_counter = 0   # Counter for early stopping

    # Lists to store loss values for graphing
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()    # Set model to training mode
        total_train_loss = 0

        for context, target in train_loader:
            optimizer.zero_grad()    # Zero the gradients before backward pass
            output = model(context)    # Get model output
            loss = loss_fn(output, target)   # Calculate loss
            loss.backward()   # Backpropagate gradients
            optimizer.step()    # Update parameters
            total_train_loss += loss.item()     # Accumulate training loss

        model.eval()    # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():   # No need to compute gradients for validation
            for context, target in val_loader:
                output = model(context)
                total_val_loss += loss_fn(output, target).item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)   

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_word2vec_model.pth")  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= 3:     # Early stopping if validation loss does not improve
                print("Early stopping.")
                break

    # Save the loss graph as "task2_loss.png"
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("task2_loss.png")  
    plt.close()

    return model

# Improved word triplet function
def find_word_triplets(model, vocab, min_similarity=0.5, max_dissimilarity=0.3, num_triplets=2):
    with torch.no_grad():
        embeddings = model.fc1.weight.detach().cpu().numpy()  # Extract word embeddings

    similarities = cosine_similarity(embeddings)   # Compute cosine similarity between word embeddings
    ignore_words = {     # List of words to ignore (common stop words)
        "[UNK]", "[PAD]", "i", "to", "at", "her", "he", "she", "it", "we", "you",
        "a", "an", "the", "of", "in", "on", "with", "and", "or", "but", "for",
        "is", "was", "has", "have", "had", "this", "that", "which", "who", "what",
        "when", "where", "why", "how", "if", "than", "then", "because", "any"
    }

    triplets = []    # List to store the word triplets
    for i in range(len(embeddings)):
        target_word = idx_to_word.get(i, "[UNK]")     # Get word corresponding to the index
        if target_word in ignore_words or target_word.startswith("##"):
            continue      # Skip ignored words or subwords

        sim_scores = similarities[i]    # Get similarity scores for the current word
        sorted_indices = np.argsort(sim_scores)[::-1]   # Sort by similarity (descending)

        # Find similar words
        similar_words = []
        for idx in sorted_indices:
            word = idx_to_word.get(idx, "[UNK]")
            if word not in ignore_words and idx != i:
                 similar_words.append(word)
            if len(similar_words) == 2:
                break

        # Find least similar (dissimilar) word
        dissimilar_word = None
        for idx in sorted_indices[::-1]:  # Reverse sorted indices for dissimilar words
            word = idx_to_word.get(idx, "[UNK]")
            if word not in ignore_words:
                dissimilar_word = word
                break
        # Add triplet if valid
        if len(similar_words) == 2 and dissimilar_word:
            triplets.append({'target': target_word, 'similar': similar_words, 'dissimilar': dissimilar_word})
        if len(triplets) >= num_triplets:     # Stop once we have enough triplets
            break

    return triplets

# Train and find triplets
embed_size = 50
trained_model = train(tokens, vocab_size, embed_size)    # Train the model
triplets = find_word_triplets(trained_model, word_to_idx)   # Find word triplets

# Display triplets
print("\nWord Triplets:")
for i, triplet in enumerate(triplets):
    print(f"Triplet {i+1}: Target: {triplet['target']}, Similar: {triplet['similar']}, Dissimilar: {triplet['dissimilar']}")
