# Language Model From Scratch (highlight) — tihassfjord

## Goal
Train a basic character-level language model for text generation using only NumPy.

## Dataset
- Any text file (sample Shakespeare text provided)
- Custom text files can be used

## Requirements
- Python 3.8+
- numpy
- matplotlib

## How to Run
```bash
# Use provided sample text
python char_lm_tihassfjord.py

# Or use your own text file
python char_lm_tihassfjord.py data/your_text.txt
```

## Example Output
```
Training char-level language model (tihassfjord).
Loaded text: 50000 characters, 65 unique
Training on character sequences...

Epoch 0, loss 3.156
Epoch 10, loss 2.234
Epoch 20, loss 1.892
...

Generated text (tihassfjord):
"The quick brown fox jumps over the lazy dog and runs through the forest..."
```

## Project Structure
```
char-lm-tihassfjord/
│
├── char_lm_tihassfjord.py         # Main language model
├── data/                          # Text data directory  
│   └── shakespeare.txt           # Sample text file
├── models/                       # Saved models
├── requirements.txt              # Dependencies
└── README.md                    # This file
```

## Key Features
- Character-level text generation
- Pure NumPy implementation
- Customizable sequence length
- Text preprocessing and tokenization
- Temperature-controlled sampling
- Model persistence
- Interactive text generation

## Learning Outcomes
- Language modeling fundamentals
- Character-level tokenization
- Sequence prediction
- Text generation techniques
- Probability distributions
- Sampling strategies

---
*Project by tihassfjord - Advanced ML Portfolio*
