"""
Simple DCGAN for MNIST by tihassfjord
Complete implementation of Deep Convolutional Generative Adversarial Network
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import imageio

class DCGAN:
    """Deep Convolutional GAN for image generation"""
    
    def __init__(self, img_shape=(28, 28, 1), latent_dim=100, learning_rate=0.0002):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        Path("generated_images").mkdir(exist_ok=True)
        
        # Build models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=optimizers.Adam(learning_rate, 0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Build and compile GAN
        self.discriminator.trainable = False
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        validity = self.discriminator(img)
        
        self.combined = models.Model(z, validity)
        self.combined.compile(
            optimizer=optimizers.Adam(learning_rate, 0.5),
            loss='binary_crossentropy'
        )
        
        # Training history
        self.d_losses = []
        self.g_losses = []
    
    def build_generator(self):
        """Build the generator network"""
        model = models.Sequential()
        
        # Dense layer to start
        model.add(layers.Dense(7 * 7 * 128, input_dim=self.latent_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Reshape((7, 7, 128)))
        
        # Upsample to 14x14
        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        # Upsample to 28x28
        model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        
        # Final layer
        model.add(layers.Conv2D(1, (3, 3), padding='same', activation='tanh'))
        
        print("Generator Architecture:")
        model.summary()
        
        return model
    
    def build_discriminator(self):
        """Build the discriminator network"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', 
                               input_shape=self.img_shape))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        
        # Hidden layers
        model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.BatchNormalization(momentum=0.8))
        
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.BatchNormalization(momentum=0.8))
        
        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        
        # Output layer
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        
        print("Discriminator Architecture:")
        model.summary()
        
        return model
    
    def load_data(self):
        """Load and preprocess MNIST data"""
        print("Loading MNIST dataset...")
        (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
        
        # Normalize to [-1, 1]
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        
        print(f"Training data shape: {X_train.shape}")
        return X_train
    
    def train(self, epochs=50, batch_size=128, save_interval=10):
        """Train the DCGAN"""
        print("Training DCGAN on MNIST (tihassfjord).")
        
        # Load data
        X_train = self.load_data()
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            
            # Generate fake images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise, verbose=0)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            
            # Store losses
            self.d_losses.append(d_loss[0])
            self.g_losses.append(g_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss[0]:.3f}, "
                      f"D Acc: {100*d_loss[1]:.1f}%, G Loss: {g_loss:.3f}")
            
            # Save generated images
            if epoch % save_interval == 0:
                self.save_generated_images(epoch)
        
        print("Training complete!")
        self.save_models()
        self.plot_training_history()
    
    def save_generated_images(self, epoch, rows=4, cols=4):
        """Generate and save sample images"""
        noise = np.random.normal(0, 1, (rows * cols, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)
        
        # Rescale images to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
        cnt = 0
        for i in range(rows):
            for j in range(cols):
                axes[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axes[i, j].axis('off')
                cnt += 1
        
        plt.suptitle(f'DCGAN MNIST - Epoch {epoch} (tihassfjord)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'generated_images/epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_images(self, num_images=64, save_individual=False):
        """Generate a batch of images"""
        print(f"Generating {num_images} images...")
        
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)
        
        # Rescale to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        # Create grid
        grid_size = int(np.sqrt(num_images))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        
        cnt = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if cnt < num_images:
                    axes[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                    axes[i, j].axis('off')
                    
                    # Save individual images if requested
                    if save_individual:
                        plt.imsave(f'generated_images/generated_{cnt:03d}.png', 
                                 gen_imgs[cnt, :, :, 0], cmap='gray')
                    cnt += 1
        
        plt.suptitle(f'Generated MNIST Digits (tihassfjord)', fontsize=16)
        plt.tight_layout()
        plt.savefig('generated_images/final_generated_grid.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return gen_imgs
    
    def create_gif(self):
        """Create animated GIF of training progression"""
        print("Creating training progression GIF...")
        
        images = []
        for epoch in range(0, len(self.d_losses), 10):
            if os.path.exists(f'generated_images/epoch_{epoch:04d}.png'):
                images.append(imageio.imread(f'generated_images/epoch_{epoch:04d}.png'))
        
        if images:
            imageio.mimsave('generated_images/training_progression.gif', images, fps=2)
            print("GIF saved: generated_images/training_progression.gif")
    
    def plot_training_history(self):
        """Plot training loss history"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.plot(self.g_losses, label='Generator Loss')
        plt.title('Training Losses (tihassfjord)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Moving average for smoother visualization
        window = 10
        if len(self.d_losses) > window:
            d_smooth = np.convolve(self.d_losses, np.ones(window)/window, mode='valid')
            g_smooth = np.convolve(self.g_losses, np.ones(window)/window, mode='valid')
            plt.plot(d_smooth, label='Discriminator (smoothed)')
            plt.plot(g_smooth, label='Generator (smoothed)')
            plt.title('Smoothed Training Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('generated_images/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save trained models"""
        self.generator.save('models/gan_generator_tihassfjord.h5')
        self.discriminator.save('models/gan_discriminator_tihassfjord.h5')
        print("Models saved successfully!")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.generator = tf.keras.models.load_model('models/gan_generator_tihassfjord.h5')
            self.discriminator = tf.keras.models.load_model('models/gan_discriminator_tihassfjord.h5')
            print("Models loaded successfully!")
            return True
        except:
            print("Could not load models. Please train first.")
            return False
    
    def interpolate_images(self, num_steps=10):
        """Generate interpolation between two random points in latent space"""
        # Generate two random points
        z1 = np.random.normal(0, 1, (1, self.latent_dim))
        z2 = np.random.normal(0, 1, (1, self.latent_dim))
        
        # Interpolate between them
        alphas = np.linspace(0, 1, num_steps)
        interpolated_z = []
        
        for alpha in alphas:
            interpolated_z.append(alpha * z2 + (1 - alpha) * z1)
        
        interpolated_z = np.vstack(interpolated_z)
        
        # Generate images
        gen_imgs = self.generator.predict(interpolated_z, verbose=0)
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        # Plot interpolation
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))
        for i in range(num_steps):
            axes[i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'{i+1}')
        
        plt.suptitle('Latent Space Interpolation (tihassfjord)', fontsize=14)
        plt.tight_layout()
        plt.savefig('generated_images/latent_interpolation.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='DCGAN for MNIST by tihassfjord')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--generate-only', action='store_true', help='Only generate images (load pre-trained model)')
    parser.add_argument('--num-generate', type=int, default=64, help='Number of images to generate')
    parser.add_argument('--create-gif', action='store_true', help='Create training progression GIF')
    
    args = parser.parse_args()
    
    # Create DCGAN instance
    dcgan = DCGAN()
    
    if args.generate_only:
        # Load model and generate images
        if dcgan.load_models():
            dcgan.generate_images(args.num_generate, save_individual=True)
            dcgan.interpolate_images()
        else:
            print("No pre-trained model found. Training first...")
            dcgan.train(epochs=args.epochs, batch_size=args.batch_size)
            dcgan.generate_images(args.num_generate, save_individual=True)
    else:
        # Train the model
        dcgan.train(epochs=args.epochs, batch_size=args.batch_size)
        
        # Generate final images
        dcgan.generate_images(args.num_generate, save_individual=True)
        dcgan.interpolate_images()
        
        # Create GIF if requested
        if args.create_gif:
            dcgan.create_gif()
    
    print(f"🎉 Image generation complete! Check the 'generated_images/' folder.")

if __name__ == "__main__":
    main()
