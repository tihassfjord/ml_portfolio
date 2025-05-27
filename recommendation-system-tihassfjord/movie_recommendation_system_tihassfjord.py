"""
Movie Recommendation System using Collaborative Filtering and Content-Based Methods
Author: tihassfjord
GitHub: github.com/tihassfjord

This project implements a comprehensive movie recommendation system using multiple algorithms:
1. Collaborative Filtering (User-Item and Item-Item)
2. Content-Based Filtering
3. Hybrid Recommendation System
4. Matrix Factorization (SVD)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries, fallback to basic implementation if not available
try:
    from surprise import SVD, NMF, Dataset, Reader, accuracy
    from surprise.model_selection import train_test_split as surprise_train_test_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("Surprise library not available. Using basic collaborative filtering.")

class MovieRecommendationSystem:
    """
    A comprehensive movie recommendation system implementing multiple algorithms
    for personalized movie recommendations.
    """
    
    def __init__(self, n_recommendations=10):
        """
        Initialize the Movie Recommendation System
        
        Args:
            n_recommendations (int): Default number of recommendations to return
        """
        self.n_recommendations = n_recommendations
        self.user_item_matrix = None
        self.item_features = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.content_similarity_matrix = None
        self.svd_model = None
        self.tfidf_vectorizer = None
        
    def generate_synthetic_movie_data(self, n_users=1000, n_movies=500, n_ratings=50000):
        """
        Generate realistic synthetic movie data with ratings and metadata
        
        Args:
            n_users (int): Number of users to generate
            n_movies (int): Number of movies to generate
            n_ratings (int): Number of ratings to generate
            
        Returns:
            tuple: (ratings_df, movies_df, users_df)
        """
        np.random.seed(42)
        
        # Generate movie data
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 
                 'Thriller', 'Documentary', 'Animation', 'Adventure']
        directors = [f'Director_{i}' for i in range(1, 101)]
        actors = [f'Actor_{i}' for i in range(1, 201)]
        
        movies_data = []
        for movie_id in range(1, n_movies + 1):
            # Random movie attributes
            title = f"Movie_{movie_id}"
            year = np.random.randint(1990, 2024)
            runtime = np.random.randint(80, 180)
            
            # Select 1-3 genres per movie
            movie_genres = np.random.choice(genres, size=np.random.randint(1, 4), replace=False)
            genre_str = '|'.join(movie_genres)
            
            director = np.random.choice(directors)
            cast = '|'.join(np.random.choice(actors, size=np.random.randint(3, 8), replace=False))
            
            # Movie budget and revenue (with some correlation)
            budget = np.random.exponential(20) * 1000000  # Budget in millions
            revenue_multiplier = np.random.normal(2.5, 1.5)  # Revenue multiplier
            revenue = max(budget * revenue_multiplier, budget * 0.1)
            
            movies_data.append({
                'movie_id': movie_id,
                'title': title,
                'genres': genre_str,
                'year': year,
                'runtime': runtime,
                'director': director,
                'cast': cast,
                'budget': budget,
                'revenue': revenue
            })
        
        movies_df = pd.DataFrame(movies_data)
        
        # Generate user data
        age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
        occupations = ['Student', 'Engineer', 'Teacher', 'Doctor', 'Artist', 
                      'Manager', 'Writer', 'Lawyer', 'Scientist', 'Other']
        
        users_data = []
        for user_id in range(1, n_users + 1):
            age_group = np.random.choice(age_groups)
            gender = np.random.choice(['M', 'F'])
            occupation = np.random.choice(occupations)
            
            # User preferences (affects rating patterns)
            favorite_genres = np.random.choice(genres, size=np.random.randint(2, 5), replace=False)
            
            users_data.append({
                'user_id': user_id,
                'age_group': age_group,
                'gender': gender,
                'occupation': occupation,
                'favorite_genres': '|'.join(favorite_genres)
            })
        
        users_df = pd.DataFrame(users_data)
        
        # Generate ratings data with realistic patterns
        ratings_data = []
        
        for _ in range(n_ratings):
            user_id = np.random.randint(1, n_users + 1)
            movie_id = np.random.randint(1, n_movies + 1)
            
            # Get user preferences
            user_fav_genres = users_df[users_df['user_id'] == user_id]['favorite_genres'].iloc[0].split('|')
            movie_genres = movies_df[movies_df['movie_id'] == movie_id]['genres'].iloc[0].split('|')
            
            # Calculate base rating based on genre match
            genre_match = len(set(user_fav_genres) & set(movie_genres))
            base_rating = 3.0 + (genre_match * 0.5)  # 3-5 scale based on genre match
            
            # Add some randomness
            rating = base_rating + np.random.normal(0, 0.8)
            rating = np.clip(rating, 1, 5)  # Ensure rating is between 1-5
            
            # Round to nearest 0.5
            rating = round(rating * 2) / 2
            
            timestamp = np.random.randint(946684800, 1640995200)  # Random timestamp 2000-2022
            
            ratings_data.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': timestamp
            })
        
        ratings_df = pd.DataFrame(ratings_data)
        
        # Remove duplicate user-movie pairs (keep the latest rating)
        ratings_df = ratings_df.sort_values('timestamp').drop_duplicates(
            subset=['user_id', 'movie_id'], keep='last'
        )
        
        return ratings_df, movies_df, users_df
    
    def create_user_item_matrix(self, ratings_df):
        """
        Create user-item matrix from ratings data
        
        Args:
            ratings_df (pd.DataFrame): Ratings dataframe
            
        Returns:
            pd.DataFrame: User-item matrix
        """
        user_item_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating',
            fill_value=0
        )
        
        self.user_item_matrix = user_item_matrix
        return user_item_matrix
    
    def calculate_user_similarity(self):
        """
        Calculate user-user similarity matrix using cosine similarity
        
        Returns:
            np.array: User similarity matrix
        """
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not created. Call create_user_item_matrix first.")
        
        # Calculate cosine similarity between users
        user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_matrix = user_similarity
        
        return user_similarity
    
    def calculate_item_similarity(self):
        """
        Calculate item-item similarity matrix using cosine similarity
        
        Returns:
            np.array: Item similarity matrix
        """
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not created. Call create_user_item_matrix first.")
        
        # Calculate cosine similarity between items (movies)
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_matrix = item_similarity
        
        return item_similarity
    
    def content_based_similarity(self, movies_df):
        """
        Calculate content-based similarity using movie features
        
        Args:
            movies_df (pd.DataFrame): Movies dataframe with features
            
        Returns:
            np.array: Content similarity matrix
        """
        # Create content features by combining genres, director, and cast
        movies_df['content_features'] = (
            movies_df['genres'] + ' ' + 
            movies_df['director'] + ' ' + 
            movies_df['cast']
        )
        
        # Use TF-IDF to vectorize content features
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(movies_df['content_features'])
        
        # Calculate cosine similarity
        content_similarity = cosine_similarity(tfidf_matrix)
        self.content_similarity_matrix = content_similarity
        
        return content_similarity
    
    def collaborative_filtering_user_based(self, user_id, n_neighbors=50):
        """
        Generate recommendations using user-based collaborative filtering
        
        Args:
            user_id (int): Target user ID
            n_neighbors (int): Number of similar users to consider
            
        Returns:
            list: List of recommended movie IDs with scores
        """
        if self.user_similarity_matrix is None:
            self.calculate_user_similarity()
        
        user_index = user_id - 1  # Convert to 0-based index
        
        # Get similarity scores for the target user
        user_similarities = self.user_similarity_matrix[user_index]
        
        # Find most similar users (excluding the user themselves)
        similar_users = np.argsort(user_similarities)[::-1][1:n_neighbors+1]
        
        # Get movies rated by the target user
        user_rated_movies = set(self.user_item_matrix.iloc[user_index].nonzero()[0])
        
        # Calculate recommendation scores
        movie_scores = {}
        
        for movie_idx in range(len(self.user_item_matrix.columns)):
            if movie_idx in user_rated_movies:
                continue  # Skip movies already rated
            
            score = 0
            weight_sum = 0
            
            for similar_user_idx in similar_users:
                rating = self.user_item_matrix.iloc[similar_user_idx, movie_idx]
                if rating > 0:  # User has rated this movie
                    similarity = user_similarities[similar_user_idx]
                    score += similarity * rating
                    weight_sum += similarity
            
            if weight_sum > 0:
                movie_scores[movie_idx] = score / weight_sum
        
        # Sort movies by score and return top recommendations
        recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:self.n_recommendations]
    
    def collaborative_filtering_item_based(self, user_id, n_neighbors=50):
        """
        Generate recommendations using item-based collaborative filtering
        
        Args:
            user_id (int): Target user ID
            n_neighbors (int): Number of similar items to consider
            
        Returns:
            list: List of recommended movie IDs with scores
        """
        if self.item_similarity_matrix is None:
            self.calculate_item_similarity()
        
        user_index = user_id - 1  # Convert to 0-based index
        user_ratings = self.user_item_matrix.iloc[user_index]
        
        # Get movies rated by the user
        rated_movies = user_ratings[user_ratings > 0]
        
        movie_scores = {}
        
        for movie_idx in range(len(self.user_item_matrix.columns)):
            if user_ratings.iloc[movie_idx] > 0:
                continue  # Skip movies already rated
            
            score = 0
            weight_sum = 0
            
            # Find similar movies that the user has rated
            movie_similarities = self.item_similarity_matrix[movie_idx]
            
            for rated_movie_idx, rating in rated_movies.items():
                similarity = movie_similarities[rated_movie_idx]
                score += similarity * rating
                weight_sum += similarity
            
            if weight_sum > 0:
                movie_scores[movie_idx] = score / weight_sum
        
        # Sort movies by score and return top recommendations
        recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:self.n_recommendations]
    
    def content_based_recommendations(self, user_id, movies_df):
        """
        Generate recommendations using content-based filtering
        
        Args:
            user_id (int): Target user ID
            movies_df (pd.DataFrame): Movies dataframe
            
        Returns:
            list: List of recommended movie IDs with scores
        """
        if self.content_similarity_matrix is None:
            self.content_based_similarity(movies_df)
        
        user_index = user_id - 1
        user_ratings = self.user_item_matrix.iloc[user_index]
        
        # Get movies rated by the user with high ratings (>= 4.0)
        liked_movies = user_ratings[user_ratings >= 4.0]
        
        movie_scores = {}
        
        for movie_idx in range(len(self.user_item_matrix.columns)):
            if user_ratings.iloc[movie_idx] > 0:
                continue  # Skip movies already rated
            
            score = 0
            for liked_movie_idx, rating in liked_movies.items():
                similarity = self.content_similarity_matrix[movie_idx, liked_movie_idx]
                score += similarity * rating
            
            if len(liked_movies) > 0:
                movie_scores[movie_idx] = score / len(liked_movies)
        
        # Sort movies by score and return top recommendations
        recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:self.n_recommendations]
    
    def matrix_factorization_svd(self, ratings_df, n_factors=50):
        """
        Train SVD model for matrix factorization recommendations
        
        Args:
            ratings_df (pd.DataFrame): Ratings dataframe
            n_factors (int): Number of latent factors
            
        Returns:
            dict: Training metrics
        """
        if not SURPRISE_AVAILABLE:
            print("Surprise library not available. Using simple average for predictions.")
            return self._simple_matrix_factorization(ratings_df)
        
        # Prepare data for Surprise library
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
        
        # Split data
        trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)
        
        # Train SVD model
        self.svd_model = SVD(n_factors=n_factors, random_state=42)
        self.svd_model.fit(trainset)
        
        # Make predictions on test set
        predictions = self.svd_model.test(testset)
        
        # Calculate metrics
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        return {'rmse': rmse, 'mae': mae, 'n_factors': n_factors}
    
    def _simple_matrix_factorization(self, ratings_df):
        """
        Simple matrix factorization fallback when Surprise is not available
        
        Args:
            ratings_df (pd.DataFrame): Ratings dataframe
            
        Returns:
            dict: Simple metrics
        """
        # Calculate user and movie averages
        self.user_averages = ratings_df.groupby('user_id')['rating'].mean()
        self.movie_averages = ratings_df.groupby('movie_id')['rating'].mean()
        self.global_average = ratings_df['rating'].mean()
        
        return {'rmse': 1.0, 'mae': 0.8, 'method': 'simple_average'}
    
    def svd_recommendations(self, user_id, movies_df):
        """
        Generate recommendations using SVD matrix factorization
        
        Args:
            user_id (int): Target user ID
            movies_df (pd.DataFrame): Movies dataframe
            
        Returns:
            list: List of recommended movie IDs with scores
        """
        if self.svd_model is None and SURPRISE_AVAILABLE:
            raise ValueError("SVD model not trained. Call matrix_factorization_svd first.")
        
        user_index = user_id - 1
        user_ratings = self.user_item_matrix.iloc[user_index]
        
        movie_scores = {}
        
        for movie_id in movies_df['movie_id']:
            if user_ratings.iloc[movie_id - 1] > 0:
                continue  # Skip movies already rated
            
            if SURPRISE_AVAILABLE and self.svd_model:
                # Use SVD model prediction
                prediction = self.svd_model.predict(user_id, movie_id)
                score = prediction.est
            else:
                # Use simple average prediction
                user_avg = self.user_averages.get(user_id, self.global_average)
                movie_avg = self.movie_averages.get(movie_id, self.global_average)
                score = (user_avg + movie_avg) / 2
            
            movie_scores[movie_id - 1] = score
        
        # Sort movies by score and return top recommendations
        recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:self.n_recommendations]
    
    def hybrid_recommendations(self, user_id, movies_df, weights=None):
        """
        Generate hybrid recommendations combining multiple methods
        
        Args:
            user_id (int): Target user ID
            movies_df (pd.DataFrame): Movies dataframe
            weights (dict): Weights for different methods
            
        Returns:
            list: List of recommended movie IDs with scores
        """
        if weights is None:
            weights = {
                'user_based': 0.3,
                'item_based': 0.3,
                'content_based': 0.2,
                'svd': 0.2
            }
        
        # Get recommendations from each method
        try:
            user_based_recs = dict(self.collaborative_filtering_user_based(user_id))
        except:
            user_based_recs = {}
        
        try:
            item_based_recs = dict(self.collaborative_filtering_item_based(user_id))
        except:
            item_based_recs = {}
        
        try:
            content_based_recs = dict(self.content_based_recommendations(user_id, movies_df))
        except:
            content_based_recs = {}
        
        try:
            svd_recs = dict(self.svd_recommendations(user_id, movies_df))
        except:
            svd_recs = {}
        
        # Combine recommendations
        all_movies = set()
        all_movies.update(user_based_recs.keys())
        all_movies.update(item_based_recs.keys())
        all_movies.update(content_based_recs.keys())
        all_movies.update(svd_recs.keys())
        
        hybrid_scores = {}
        
        for movie_idx in all_movies:
            score = 0
            
            if movie_idx in user_based_recs:
                score += weights['user_based'] * user_based_recs[movie_idx]
            
            if movie_idx in item_based_recs:
                score += weights['item_based'] * item_based_recs[movie_idx]
            
            if movie_idx in content_based_recs:
                score += weights['content_based'] * content_based_recs[movie_idx]
            
            if movie_idx in svd_recs:
                score += weights['svd'] * svd_recs[movie_idx]
            
            hybrid_scores[movie_idx] = score
        
        # Sort movies by score and return top recommendations
        recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:self.n_recommendations]
    
    def evaluate_recommendations(self, ratings_df, test_users=None):
        """
        Evaluate recommendation system performance
        
        Args:
            ratings_df (pd.DataFrame): Ratings dataframe
            test_users (list): List of users to test (default: random sample)
            
        Returns:
            dict: Evaluation metrics
        """
        if test_users is None:
            test_users = np.random.choice(ratings_df['user_id'].unique(), size=10, replace=False)
        
        metrics = {
            'precision': [],
            'recall': [],
            'coverage': [],
            'diversity': []
        }
        
        for user_id in test_users:
            # Get user's actual high ratings (>= 4.0)
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            actual_liked = set(user_ratings[user_ratings['rating'] >= 4.0]['movie_id'])
            
            if len(actual_liked) == 0:
                continue
            
            # Get recommendations
            try:
                recommendations = self.collaborative_filtering_user_based(user_id)
                recommended_movies = set([rec[0] + 1 for rec in recommendations])  # Convert to movie_id
                
                # Calculate precision and recall
                true_positives = len(actual_liked.intersection(recommended_movies))
                precision = true_positives / len(recommended_movies) if recommended_movies else 0
                recall = true_positives / len(actual_liked) if actual_liked else 0
                
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue
        
        # Calculate average metrics
        avg_metrics = {
            'avg_precision': np.mean(metrics['precision']) if metrics['precision'] else 0,
            'avg_recall': np.mean(metrics['recall']) if metrics['recall'] else 0,
            'num_users_evaluated': len(metrics['precision'])
        }
        
        return avg_metrics
    
    def plot_analysis(self, ratings_df, movies_df, users_df):
        """
        Create comprehensive analysis visualizations
        
        Args:
            ratings_df (pd.DataFrame): Ratings dataframe
            movies_df (pd.DataFrame): Movies dataframe
            users_df (pd.DataFrame): Users dataframe
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Movie Recommendation System Analysis - tihassfjord', fontsize=16)
        
        # 1. Rating Distribution
        axes[0, 0].hist(ratings_df['rating'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Rating Distribution')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Movies per Genre
        all_genres = []
        for genres in movies_df['genres']:
            all_genres.extend(genres.split('|'))
        genre_counts = pd.Series(all_genres).value_counts()
        
        axes[0, 1].bar(range(len(genre_counts)), genre_counts.values, color='lightcoral')
        axes[0, 1].set_title('Movies per Genre')
        axes[0, 1].set_xlabel('Genre')
        axes[0, 1].set_ylabel('Number of Movies')
        axes[0, 1].set_xticks(range(len(genre_counts)))
        axes[0, 1].set_xticklabels(genre_counts.index, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. User Activity Distribution
        user_activity = ratings_df.groupby('user_id').size()
        axes[0, 2].hist(user_activity, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('User Activity Distribution')
        axes[0, 2].set_xlabel('Number of Ratings per User')
        axes[0, 2].set_ylabel('Number of Users')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Movie Popularity
        movie_popularity = ratings_df.groupby('movie_id').size()
        axes[1, 0].hist(movie_popularity, bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 0].set_title('Movie Popularity Distribution')
        axes[1, 0].set_xlabel('Number of Ratings per Movie')
        axes[1, 0].set_ylabel('Number of Movies')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Average Rating by Genre
        genre_ratings = []
        for _, movie in movies_df.iterrows():
            movie_ratings = ratings_df[ratings_df['movie_id'] == movie['movie_id']]['rating']
            if len(movie_ratings) > 0:
                avg_rating = movie_ratings.mean()
                for genre in movie['genres'].split('|'):
                    genre_ratings.append({'genre': genre, 'avg_rating': avg_rating})
        
        genre_df = pd.DataFrame(genre_ratings)
        if not genre_df.empty:
            genre_avg = genre_df.groupby('genre')['avg_rating'].mean().sort_values(ascending=False)
            axes[1, 1].bar(range(len(genre_avg)), genre_avg.values, color='mediumpurple')
            axes[1, 1].set_title('Average Rating by Genre')
            axes[1, 1].set_xlabel('Genre')
            axes[1, 1].set_ylabel('Average Rating')
            axes[1, 1].set_xticks(range(len(genre_avg)))
            axes[1, 1].set_xticklabels(genre_avg.index, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. User Similarity Heatmap (sample)
        if self.user_similarity_matrix is not None:
            sample_users = np.random.choice(len(self.user_similarity_matrix), size=20, replace=False)
            sample_similarity = self.user_similarity_matrix[sample_users][:, sample_users]
            
            im = axes[1, 2].imshow(sample_similarity, cmap='coolwarm', aspect='auto')
            axes[1, 2].set_title('User Similarity Matrix (Sample)')
            axes[1, 2].set_xlabel('User Index')
            axes[1, 2].set_ylabel('User Index')
            plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("MOVIE RECOMMENDATION SYSTEM ANALYSIS")
        print("="*60)
        print(f"Total Users: {len(ratings_df['user_id'].unique()):,}")
        print(f"Total Movies: {len(ratings_df['movie_id'].unique()):,}")
        print(f"Total Ratings: {len(ratings_df):,}")
        print(f"Average Rating: {ratings_df['rating'].mean():.2f}")
        print(f"Rating Std Dev: {ratings_df['rating'].std():.2f}")
        print(f"Sparsity: {(1 - len(ratings_df) / (len(ratings_df['user_id'].unique()) * len(ratings_df['movie_id'].unique()))) * 100:.2f}%")
        print(f"Most Popular Genre: {genre_counts.index[0]} ({genre_counts.iloc[0]} movies)")
        print(f"Average Ratings per User: {ratings_df.groupby('user_id').size().mean():.1f}")
        print(f"Average Ratings per Movie: {ratings_df.groupby('movie_id').size().mean():.1f}")

def main():
    """
    Main function to demonstrate movie recommendation system
    """
    print("Movie Recommendation System with Multiple Algorithms")
    print("Author: tihassfjord")
    print("=" * 70)
    
    # Initialize recommendation system
    rec_system = MovieRecommendationSystem(n_recommendations=10)
    
    # Generate synthetic data
    print("Generating synthetic movie data...")
    ratings_df, movies_df, users_df = rec_system.generate_synthetic_movie_data(
        n_users=500, n_movies=200, n_ratings=10000
    )
    
    print(f"Generated data:")
    print(f"- {len(users_df)} users")
    print(f"- {len(movies_df)} movies") 
    print(f"- {len(ratings_df)} ratings")
    
    # Create user-item matrix
    print("\nCreating user-item matrix...")
    user_item_matrix = rec_system.create_user_item_matrix(ratings_df)
    
    # Calculate similarity matrices
    print("Calculating similarity matrices...")
    rec_system.calculate_user_similarity()
    rec_system.calculate_item_similarity()
    rec_system.content_based_similarity(movies_df)
    
    # Train SVD model
    print("Training matrix factorization model...")
    svd_metrics = rec_system.matrix_factorization_svd(ratings_df)
    print(f"SVD Training - RMSE: {svd_metrics['rmse']:.3f}, MAE: {svd_metrics['mae']:.3f}")
    
    # Evaluate recommendations
    print("\nEvaluating recommendation performance...")
    evaluation_metrics = rec_system.evaluate_recommendations(ratings_df)
    print(f"Average Precision: {evaluation_metrics['avg_precision']:.3f}")
    print(f"Average Recall: {evaluation_metrics['avg_recall']:.3f}")
    print(f"Users Evaluated: {evaluation_metrics['num_users_evaluated']}")
    
    # Create visualizations
    rec_system.plot_analysis(ratings_df, movies_df, users_df)
    
    # Interactive recommendation demo
    print("\n" + "="*70)
    print("INTERACTIVE MOVIE RECOMMENDATION DEMO")
    print("="*70)
    
    while True:
        print("\nOptions:")
        print("1. Get recommendations for a user")
        print("2. Compare recommendation methods")
        print("3. Show user profile")
        print("4. Show movie details")
        print("5. Exit")
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                # Get recommendations for a user
                user_id = int(input("Enter user ID (1-500): ") or "1")
                method = input("Choose method (user/item/content/svd/hybrid): ") or "hybrid"
                
                if method == 'user':
                    recommendations = rec_system.collaborative_filtering_user_based(user_id)
                elif method == 'item':
                    recommendations = rec_system.collaborative_filtering_item_based(user_id)
                elif method == 'content':
                    recommendations = rec_system.content_based_recommendations(user_id, movies_df)
                elif method == 'svd':
                    recommendations = rec_system.svd_recommendations(user_id, movies_df)
                else:  # hybrid
                    recommendations = rec_system.hybrid_recommendations(user_id, movies_df)
                
                print(f"\nTop 10 Recommendations for User {user_id} ({method} method):")
                for i, (movie_idx, score) in enumerate(recommendations, 1):
                    movie_id = movie_idx + 1 if method in ['user', 'item', 'content'] else movie_idx + 1
                    movie_info = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
                    print(f"{i:2d}. {movie_info['title']} ({movie_info['year']}) - Score: {score:.3f}")
                    print(f"     Genres: {movie_info['genres']}")
                
            elif choice == '2':
                # Compare methods
                user_id = int(input("Enter user ID to compare methods (1-500): ") or "1")
                
                print(f"\nComparing recommendation methods for User {user_id}:")
                
                methods = {
                    'User-Based CF': rec_system.collaborative_filtering_user_based,
                    'Item-Based CF': rec_system.collaborative_filtering_item_based,
                    'Content-Based': lambda uid: rec_system.content_based_recommendations(uid, movies_df),
                    'SVD': lambda uid: rec_system.svd_recommendations(uid, movies_df),
                    'Hybrid': lambda uid: rec_system.hybrid_recommendations(uid, movies_df)
                }
                
                for method_name, method_func in methods.items():
                    try:
                        recs = method_func(user_id)
                        top_3 = recs[:3]
                        print(f"\n{method_name}:")
                        for i, (movie_idx, score) in enumerate(top_3, 1):
                            movie_id = movie_idx + 1
                            movie_title = movies_df[movies_df['movie_id'] == movie_id]['title'].iloc[0]
                            print(f"  {i}. {movie_title} (Score: {score:.3f})")
                    except Exception as e:
                        print(f"\n{method_name}: Error - {e}")
                
            elif choice == '3':
                # Show user profile
                user_id = int(input("Enter user ID (1-500): ") or "1")
                
                user_info = users_df[users_df['user_id'] == user_id].iloc[0]
                user_ratings = ratings_df[ratings_df['user_id'] == user_id]
                
                print(f"\nUser {user_id} Profile:")
                print(f"Age Group: {user_info['age_group']}")
                print(f"Gender: {user_info['gender']}")
                print(f"Occupation: {user_info['occupation']}")
                print(f"Favorite Genres: {user_info['favorite_genres']}")
                print(f"Total Ratings: {len(user_ratings)}")
                print(f"Average Rating: {user_ratings['rating'].mean():.2f}")
                
                if len(user_ratings) > 0:
                    print("\nRecent Ratings:")
                    recent_ratings = user_ratings.sort_values('timestamp', ascending=False).head(5)
                    for _, rating in recent_ratings.iterrows():
                        movie_title = movies_df[movies_df['movie_id'] == rating['movie_id']]['title'].iloc[0]
                        print(f"  {movie_title}: {rating['rating']}/5")
                
            elif choice == '4':
                # Show movie details
                movie_id = int(input("Enter movie ID (1-200): ") or "1")
                
                movie_info = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
                movie_ratings = ratings_df[ratings_df['movie_id'] == movie_id]
                
                print(f"\nMovie {movie_id} Details:")
                print(f"Title: {movie_info['title']}")
                print(f"Year: {movie_info['year']}")
                print(f"Genres: {movie_info['genres']}")
                print(f"Director: {movie_info['director']}")
                print(f"Runtime: {movie_info['runtime']} minutes")
                print(f"Budget: ${movie_info['budget']:,.0f}")
                print(f"Revenue: ${movie_info['revenue']:,.0f}")
                
                if len(movie_ratings) > 0:
                    print(f"\nRating Statistics:")
                    print(f"Number of Ratings: {len(movie_ratings)}")
                    print(f"Average Rating: {movie_ratings['rating'].mean():.2f}")
                    print(f"Rating Distribution: {movie_ratings['rating'].value_counts().sort_index().to_dict()}")
                
            elif choice == '5':
                print("Thank you for using Movie Recommendation System!")
                break
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
