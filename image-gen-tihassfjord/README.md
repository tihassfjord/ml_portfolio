# Image Generation System (highlight) — tihassfjord

## Goal
Generate synthetic images using GANs (Generative Adversarial Networks) on MNIST dataset.

## Dataset
- MNIST (default, can be extended to other datasets)
- Generated synthetic digit images

## Requirements
- Python 3.8+
- tensorflow or torch
- numpy
- matplotlib
- PIL

## How to Run
```bash
# Train GAN and generate images
python gan_mnist_tihassfjord.py

# Generate images from saved model
python gan_mnist_tihassfjord.py --generate-only
```

## Example Output
```
Training DCGAN on MNIST (tihassfjord).
Epoch 0/50, D Loss: 1.234, G Loss: 0.876
Epoch 10/50, D Loss: 0.567, G Loss: 1.123
...
Training complete! Generated 64 sample images.
Model saved: models/gan_generator_tihassfjord.h5
```

## Project Structure
```
image-gen-tihassfjord/
│
├── gan_mnist_tihassfjord.py          # Main GAN implementation
├── models/                           # Saved models
├── generated_images/                 # Output images
├── requirements.txt                  # Dependencies
└── README.md                        # This file
```

## Key Features
- Deep Convolutional GAN (DCGAN) architecture
- Progressive training with loss monitoring
- High-quality image generation
- Model persistence and loading
- Batch image generation
- Customizable network architecture
- Training visualization

## Learning Outcomes
- Generative Adversarial Networks
- Deep learning with TensorFlow/PyTorch
- Image generation techniques
- GAN training strategies
- Computer vision fundamentals
- Neural network architectures

---
*Project by tihassfjord - Advanced ML Portfolio*
