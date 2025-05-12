import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from math import exp
from task1 import WordPieceTokenizer   
from task2 import Word2VecModel      

# NeuralLMDataset Definition

class NeuralLMDataset(Dataset):
    def __init__(self, corpus_file, vocab, context_size=3):
        self.vocab = vocab
        self.context_size = context_size
        self.tokenizer = WordPieceTokenizer(vocab_size=1000)
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read()
        # Preprocess data: lowercase, tokenize, etc.
        tokens = self.tokenizer.preprocess_data(corpus)
        self.indexed_tokens = [vocab.get(token, vocab["[UNK]"]) for token in tokens]
        self.data = []
        for i in range(len(self.indexed_tokens) - context_size):
            context = self.indexed_tokens[i:i+context_size]
            target = self.indexed_tokens[i+context_size]
            self.data.append((torch.tensor(context, dtype=torch.long),
                              torch.tensor(target, dtype=torch.long)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Neural LM Architectures

class NeuralLM1(nn.Module):
    def __init__(self, vocab_size, embedding_dim=150, hidden_dim=256, context_size=3):
        super(NeuralLM1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embeds = self.embedding(x)                # [batch, context_size, embedding_dim]
        flatten = embeds.view(embeds.shape[0], -1)  # [batch, context_size * embedding_dim]
        out = torch.relu(self.fc1(flatten))
        return self.fc2(out)

class NeuralLM2(nn.Module):
    def __init__(self, vocab_size, embedding_dim=150, hidden_dim=256, context_size=3):
        super(NeuralLM2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, vocab_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        embeds = self.embedding(x)
        flatten = embeds.view(embeds.shape[0], -1)
        out = torch.relu(self.fc1(flatten))
        out = torch.relu(self.fc2(out))
        out = self.dropout(out)
        return self.fc3(out)

class NeuralLM3(nn.Module):
    def __init__(self, vocab_size, embedding_dim=150, hidden_dim=256, context_size=3):
        super(NeuralLM3, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.fc_mid = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, x):
        embeds = self.embedding(x)
        flatten = embeds.view(embeds.shape[0], -1)
        out = torch.tanh(self.fc1(flatten))
        out = self.dropout(out)
        out = torch.relu(self.fc_mid(out))
        return self.fc2(out)

# Accuracy and Perplexity Functions
def compute_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def compute_perplexity(loss):
    return exp(loss)

# Training Function for Neural LM Models
def train_model(model, train_loader, val_loader, model_name, epochs=10, lr=0.0005):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    train_losses, val_losses = [], []
    train_acc, val_acc = [], []
    train_perp, val_perp = [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        total_batches = 0
        
        for context, target in train_loader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += compute_accuracy(output, target)
            total_batches += 1
            
        avg_train_loss = total_loss / total_batches
        avg_train_acc = total_acc / total_batches
        train_losses.append(avg_train_loss)
        train_perp.append(compute_perplexity(avg_train_loss))
        train_acc.append(avg_train_acc)
        
        model.eval()
        total_loss = 0
        total_acc = 0
        total_batches = 0
        
        with torch.no_grad():
            for context, target in val_loader:
                output = model(context)
                loss = criterion(output, target)
                total_loss += loss.item()
                total_acc += compute_accuracy(output, target)
                total_batches += 1
                
        avg_val_loss = total_loss / total_batches
        avg_val_acc = total_acc / total_batches
        val_losses.append(avg_val_loss)
        val_perp.append(compute_perplexity(avg_val_loss))
        val_acc.append(avg_val_acc)
        
        scheduler.step(avg_train_loss)
        
        print(f"{model_name} Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perp[-1]:.4f}, Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perp[-1]:.4f}, Val Acc: {avg_val_acc:.4f}")
    
    plt.figure()
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epochs for {model_name}")
    plt.legend()
    plt.savefig(f"{model_name}_loss.png")
    plt.show()
    
    return model

# Prediction Pipeline for Neural LM Models
def predict_next_tokens(models, sentence, tokenizer, vocab, idx_to_word, context_size=3, num_tokens=3):
    for model in models:
        model.eval()
    tokens = tokenizer.preprocess_data(sentence)
    tokens = tokens[-context_size:]
    context_indices = [vocab.get(token, vocab["[UNK]"]) for token in tokens]
    context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0)
    
    predicted_tokens = []
    with torch.no_grad():
        for _ in range(num_tokens):
            logits = [model(context_tensor) for model in models]
            avg_logits = torch.mean(torch.stack(logits), dim=0)
            predicted_idx = torch.argmax(avg_logits, dim=1).item()
            predicted_word = idx_to_word.get(predicted_idx, "[UNK]")
            predicted_tokens.append(predicted_word)
            context_indices = context_indices[1:] + [predicted_idx]
            context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0)
    return predicted_tokens


# Main Pipeline for Neural LM Training & Testing

if __name__ == "__main__":
    print("starting the algorithm...")
    # Load corpus from corpus.txt
    corpus_file = "corpus.txt"
    
    # Build a vocabulary from corpus tokens with fallback tokens
    tokenizer = WordPieceTokenizer(vocab_size=1000)
    corpus_text = open(corpus_file, 'r', encoding='utf-8').read()
    tokenizer.train(corpus_text)
    tokens = tokenizer.preprocess_data(corpus_text)
    unique_tokens = list(set(tokens))
    if "[PAD]" not in unique_tokens:
        unique_tokens.append("[PAD]")
    if "[UNK]" not in unique_tokens:
        unique_tokens.append("[UNK]")
    vocab = {word: idx for idx, word in enumerate(unique_tokens)}
    vocab_size = len(vocab)
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    # Initialize NeuralLMDataset and DataLoaders
    dataset = NeuralLMDataset(corpus_file, vocab, context_size=3)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Load pretrained Word2Vec embeddings (from Task2 checkpoint)
    pretrained_embeddings = None
    try:
        word2vec = Word2VecModel(vocab_size, embed_size=150)
        word2vec.load_state_dict(torch.load("best_word2vec_model.pth", map_location=torch.device('cpu')))
        # Extract embeddings: shape (embed_size, vocab_size), then transpose to (vocab_size, embed_size)
        pretrained_embeddings = word2vec.fc1.weight.detach().cpu().numpy().T
    except Exception as e:
        print(f"Failed to load Word2Vec model: {e}")
        pretrained_embeddings = None

    # Train three Neural LM variations: NeuralLM1, NeuralLM2, NeuralLM3
    trained_models = []
    for model_class, name in zip([NeuralLM1, NeuralLM2, NeuralLM3],
                                 ["NeuralLM1", "NeuralLM2", "NeuralLM3"]):
        print(f"\nTraining {name} ...")
        model = model_class(vocab_size, embedding_dim=150, hidden_dim=256, context_size=3)
        if pretrained_embeddings is not None:
            model.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))
        trained_model = train_model(model, train_loader, val_loader, name, epochs=10, lr=0.0005)
        trained_models.append(trained_model)
    
    # Prediction on test data from test.txt
    with open("test.txt", 'r', encoding='utf-8') as f:
        test_sentences = [line.strip() for line in f if line.strip()]
    
    print("\nTest Predictions (using ensemble of all three models):")
    for sentence in test_sentences:
        predicted = predict_next_tokens(trained_models, sentence, tokenizer, vocab, idx_to_word, context_size=3, num_tokens=3)
        print(f"Input: {sentence}")
        print(f"Predicted Next Tokens: {' '.join(predicted)}\n")