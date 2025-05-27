"""
Simple Character-level Language Model by tihassfjord
Implements text generation using only NumPy
"""

import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

class CharacterLM:
    """Character-level Language Model from scratch"""
    
    def __init__(self, sequence_length=25, learning_rate=0.1):
        self.seq_length = sequence_length
        self.learning_rate = learning_rate
        self.chars = None
        self.char_to_ix = {}
        self.ix_to_char = {}
        self.vocab_size = 0
        self.W = None
        self.losses = []
        
        # Create directories
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
    
    def load_text(self, file_path=None):
        """Load and preprocess text data"""
        if file_path is None:
            # Create sample text if none provided
            self._create_sample_text()
            file_path = "data/shakespeare.txt"
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        print(f"Loading text from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Get unique characters
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create mappings
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f"Loaded text: {len(text)} characters, {self.vocab_size} unique")
        print(f"Vocabulary: {self.chars}")
        
        return text
    
    def _create_sample_text(self):
        """Create sample Shakespeare-like text"""
        sample_text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them? To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action.
""" * 10  # Repeat to have more text
        
        with open("data/shakespeare.txt", "w", encoding='utf-8') as f:
            f.write(sample_text)
        
        print("Created sample text: data/shakespeare.txt")
    
    def prepare_data(self, text):
        """Prepare training data"""
        # Convert text to indices
        data = [self.char_to_ix[ch] for ch in text]
        
        # Create input-output pairs
        X, Y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            Y.append(data[i + self.seq_length])
        
        return np.array(X), np.array(Y)
    
    def initialize_weights(self):
        """Initialize model weights"""
        # Weight matrix: maps input character to output character probabilities
        self.W = np.random.randn(self.vocab_size, self.vocab_size) * 0.01
        print(f"Initialized weights: {self.W.shape}")
    
    def forward(self, X):
        """Forward pass"""
        # Simple approach: average the character representations
        # For each sequence, take the mean of one-hot vectors
        batch_size = X.shape[0]
        
        # Convert indices to one-hot
        X_onehot = np.zeros((batch_size, self.seq_length, self.vocab_size))
        for i in range(batch_size):
            for j in range(self.seq_length):
                X_onehot[i, j, X[i, j]] = 1
        
        # Average over sequence length
        X_avg = np.mean(X_onehot, axis=1)  # (batch_size, vocab_size)
        
        # Compute logits
        logits = X_avg @ self.W  # (batch_size, vocab_size)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs, X_avg
    
    def compute_loss(self, probs, Y):
        """Compute cross-entropy loss"""
        batch_size = len(Y)
        # Avoid log(0)
        log_probs = np.log(probs + 1e-8)
        loss = -np.mean(log_probs[np.arange(batch_size), Y])
        return loss
    
    def backward(self, probs, Y, X_avg):
        """Backward pass"""
        batch_size = len(Y)
        
        # Gradient of loss w.r.t. logits
        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), Y] -= 1
        grad_logits /= batch_size
        
        # Gradient w.r.t. weights
        grad_W = X_avg.T @ grad_logits
        
        return grad_W
    
    def train(self, text, epochs=100, batch_size=32):
        """Train the language model"""
        print("Training char-level language model (tihassfjord).")
        
        # Prepare data
        X, Y = self.prepare_data(text)
        print(f"Training sequences: {len(X)}")
        
        # Initialize weights
        self.initialize_weights()
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_Y = Y[i:i + batch_size]
                
                if len(batch_X) == 0:
                    continue
                
                # Forward pass
                probs, X_avg = self.forward(batch_X)
                
                # Compute loss
                loss = self.compute_loss(probs, batch_Y)
                epoch_loss += loss
                num_batches += 1
                
                # Backward pass
                grad_W = self.backward(probs, batch_Y, X_avg)
                
                # Update weights
                self.W -= self.learning_rate * grad_W
            
            # Average loss for epoch
            avg_loss = epoch_loss / max(num_batches, 1)
            self.losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, loss {avg_loss:.3f}")
                
                # Generate sample text
                if epoch % 20 == 0:
                    sample = self.generate(length=100, temperature=0.8)
                    print(f"Sample: {sample[:50]}...")
    
    def generate(self, seed_text=None, length=200, temperature=1.0):
        """Generate text using the trained model"""
        if seed_text is None:
            # Start with random character
            seed_text = self.ix_to_char[np.random.randint(self.vocab_size)]
        
        # Ensure seed is long enough
        while len(seed_text) < self.seq_length:
            seed_text = seed_text + seed_text
        
        generated = seed_text[-self.seq_length:]
        
        for _ in range(length):
            # Convert current sequence to indices
            seq_indices = [self.char_to_ix.get(ch, 0) for ch in generated[-self.seq_length:]]
            
            # Predict next character
            X_input = np.array([seq_indices])
            probs, _ = self.forward(X_input)
            probs = probs[0]
            
            # Apply temperature
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)
            
            # Sample next character
            next_ix = np.random.choice(self.vocab_size, p=probs)
            next_char = self.ix_to_char[next_ix]
            
            generated += next_char
        
        return generated[self.seq_length:]  # Remove seed
    
    def plot_loss(self):
        """Plot training loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title('Training Loss - Character Language Model (tihassfjord)')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.grid(True)
        plt.show()
    
    def save_model(self, filename="char_lm_tihassfjord.npz"):
        """Save the trained model"""
        model_path = f"models/{filename}"
        np.savez(model_path,
                 W=self.W,
                 chars=self.chars,
                 seq_length=self.seq_length)
        
        # Save metadata
        metadata = {
            'vocab_size': self.vocab_size,
            'seq_length': self.seq_length,
            'learning_rate': self.learning_rate,
            'char_to_ix': self.char_to_ix,
            'ix_to_char': {str(k): v for k, v in self.ix_to_char.items()}
        }
        
        with open("models/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved: {model_path}")
    
    def load_model(self, filename="char_lm_tihassfjord.npz"):
        """Load a trained model"""
        model_path = f"models/{filename}"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load weights
        data = np.load(model_path)
        self.W = data['W']
        self.chars = data['chars'].tolist()
        self.seq_length = int(data['seq_length'])
        
        # Load metadata
        with open("models/metadata.json", "r") as f:
            metadata = json.load(f)
        
        self.vocab_size = metadata['vocab_size']
        self.char_to_ix = metadata['char_to_ix']
        self.ix_to_char = {int(k): v for k, v in metadata['ix_to_char'].items()}
        
        print(f"Model loaded: {model_path}")

def interactive_generation(model):
    """Interactive text generation"""
    print("\n" + "="*50)
    print("INTERACTIVE TEXT GENERATION")
    print("="*50)
    print("Commands:")
    print("  'generate' or 'g' - Generate text")
    print("  'quit' or 'q' - Exit")
    print("  'help' or 'h' - Show this help")
    
    while True:
        cmd = input("\n> ").strip().lower()
        
        if cmd in ['quit', 'q']:
            break
        elif cmd in ['help', 'h']:
            print("Commands: generate/g, quit/q, help/h")
        elif cmd in ['generate', 'g']:
            try:
                seed = input("Enter seed text (or press Enter for random): ").strip()
                if not seed:
                    seed = None
                
                length = input("Enter length (default 200): ").strip()
                length = int(length) if length else 200
                
                temp = input("Enter temperature 0.1-2.0 (default 1.0): ").strip()
                temp = float(temp) if temp else 1.0
                
                print("\nGenerating text...")
                generated = model.generate(seed, length, temp)
                print(f"\nGenerated text:\n{generated}")
                
            except (ValueError, KeyboardInterrupt):
                print("Invalid input or interrupted.")
        else:
            print("Unknown command. Type 'help' for commands.")

def main():
    """Main function"""
    # Get text file from command line argument
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        # Create language model
        model = CharacterLM(sequence_length=25, learning_rate=0.1)
        
        # Load and train
        text = model.load_text(file_path)
        model.train(text, epochs=50, batch_size=32)
        
        # Plot training progress
        model.plot_loss()
        
        # Generate sample text
        print("\nGenerating sample text...")
        for temp in [0.5, 1.0, 1.5]:
            print(f"\nTemperature {temp}:")
            sample = model.generate(length=200, temperature=temp)
            print(sample)
        
        # Save model
        model.save_model()
        
        # Interactive generation
        interactive_generation(model)
        
        print("\nLanguage model training complete! (tihassfjord)")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required packages installed.")
        print("Install with: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
