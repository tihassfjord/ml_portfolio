# Movie Recommendation System

**Author:** tihassfjord  
**GitHub:** [github.com/tihassfjord](https://github.com/tihassfjord)

## Overview

This project implements a comprehensive movie recommendation system using multiple state-of-the-art algorithms including collaborative filtering, content-based filtering, and matrix factorization techniques. The system demonstrates advanced machine learning concepts in recommendation engines with realistic synthetic data generation.

## Features

### 🎬 Multiple Recommendation Algorithms
- **User-Based Collaborative Filtering** - Find similar users and recommend their preferences
- **Item-Based Collaborative Filtering** - Recommend movies similar to user's liked items
- **Content-Based Filtering** - Recommend based on movie features (genres, cast, director)
- **Matrix Factorization (SVD)** - Advanced latent factor modeling
- **Hybrid Recommendation** - Combine multiple methods for better accuracy

### 📊 Advanced Data Generation
- **Realistic Movie Database** with genres, cast, directors, budget, revenue
- **User Profiles** with demographics and preference patterns
- **Intelligent Rating Generation** based on user-movie compatibility
- **Market Simulation** with realistic rating distributions

### 🔍 Comprehensive Analysis
- **Rating Pattern Analysis** and user behavior insights
- **Genre Popularity** and rating correlation studies
- **User Similarity Heatmaps** and clustering visualization
- **Recommendation Performance Metrics** (Precision, Recall, Coverage)

### 🎯 Interactive Features
- **Real-time Recommendations** for any user
- **Method Comparison** to evaluate different algorithms
- **User Profile Analysis** with rating history
- **Movie Details** with comprehensive metadata

## Technical Architecture

### Data Models
```
Users: user_id, age_group, gender, occupation, favorite_genres
Movies: movie_id, title, genres, year, runtime, director, cast, budget, revenue
Ratings: user_id, movie_id, rating, timestamp
```

### Recommendation Algorithms

#### 1. Collaborative Filtering
- **User-Based:** Cosine similarity between user rating vectors
- **Item-Based:** Cosine similarity between movie rating profiles
- **Neighborhood Size:** Configurable top-K similar users/items

#### 2. Content-Based Filtering
- **TF-IDF Vectorization** of movie features (genres + cast + director)
- **Cosine Similarity** for content matching
- **User Profile Building** from highly-rated movies (≥4.0)

#### 3. Matrix Factorization
- **SVD (Singular Value Decomposition)** with Surprise library
- **Latent Factor Learning** (50 dimensions default)
- **Regularization** to prevent overfitting

#### 4. Hybrid System
- **Weighted Combination** of all methods
- **Default Weights:** User-Based (30%), Item-Based (30%), Content (20%), SVD (20%)
- **Customizable Weighting** for different scenarios

## Installation & Setup

### Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
# Optional for advanced features:
pip install scikit-surprise
```

### Quick Start
```bash
cd recommendation-system-tihassfjord
python movie_recommendation_system_tihassfjord.py
```

## Usage Examples

### Basic Recommendation
```python
from movie_recommendation_system_tihassfjord import MovieRecommendationSystem

# Initialize system
rec_system = MovieRecommendationSystem(n_recommendations=10)

# Generate synthetic data
ratings_df, movies_df, users_df = rec_system.generate_synthetic_movie_data(
    n_users=500, n_movies=200, n_ratings=10000
)

# Train models
rec_system.create_user_item_matrix(ratings_df)
rec_system.calculate_user_similarity()
rec_system.matrix_factorization_svd(ratings_df)

# Get recommendations
recommendations = rec_system.hybrid_recommendations(user_id=1, movies_df=movies_df)
```

### Compare Methods
```python
user_id = 42
methods = {
    'User-Based': rec_system.collaborative_filtering_user_based(user_id),
    'Item-Based': rec_system.collaborative_filtering_item_based(user_id),
    'Content-Based': rec_system.content_based_recommendations(user_id, movies_df),
    'SVD': rec_system.svd_recommendations(user_id, movies_df),
    'Hybrid': rec_system.hybrid_recommendations(user_id, movies_df)
}
```

## Performance Metrics

### Evaluation Methods
- **Precision:** Relevant recommendations / Total recommendations
- **Recall:** Relevant recommendations / Total relevant items  
- **Coverage:** Percentage of items that can be recommended
- **Diversity:** Variety in recommendation categories

### Expected Performance
- **User-Based CF:** High precision for users with many ratings
- **Item-Based CF:** Good for discovering similar movies
- **Content-Based:** Excellent for new users (cold start)
- **SVD:** Best overall accuracy with sufficient data
- **Hybrid:** Combines strengths of all methods

## Data Generation Details

### Movie Features
- **Genres:** 10 categories (Action, Comedy, Drama, etc.)
- **Directors:** 100 unique directors
- **Cast:** 200 unique actors, 3-8 per movie
- **Financial Data:** Realistic budget-revenue correlations
- **Temporal Data:** Movies from 1990-2024

### User Simulation
- **Demographics:** Age groups, gender, occupation
- **Preferences:** 2-5 favorite genres per user
- **Rating Behavior:** Genre preference influences ratings
- **Realistic Distribution:** Follows typical rating patterns

### Rating Generation Algorithm
```python
base_rating = 3.0 + (genre_match_count * 0.5)
final_rating = base_rating + normal_noise(μ=0, σ=0.8)
rating = clip(round(final_rating * 2) / 2, 1, 5)  # Round to 0.5
```

## Visualizations

### 1. Data Analysis Dashboard
- **Rating Distribution** - Overall system rating patterns
- **Genre Popularity** - Movies per genre and average ratings
- **User Activity** - Rating frequency distribution
- **Movie Popularity** - Number of ratings per movie

### 2. System Performance
- **User Similarity Heatmap** - Visualize user clustering
- **Recommendation Accuracy** - Precision/Recall curves
- **Method Comparison** - Side-by-side algorithm performance

### 3. Interactive Exploration
- **User Profiles** - Individual user rating history
- **Movie Details** - Comprehensive movie information
- **Recommendation Lists** - Top-K recommendations with scores

## Advanced Features

### Cold Start Solutions
- **New Users:** Content-based recommendations from demographic data
- **New Movies:** Content similarity to existing catalog
- **Popularity Fallback:** Most popular items in relevant categories

### Scalability Considerations
- **Sparse Matrix Operations** for memory efficiency
- **Incremental Learning** for real-time updates
- **Batch Processing** for large-scale recommendation generation

### Quality Assurance
- **Data Validation** ensures realistic rating patterns
- **Recommendation Diversity** prevents filter bubbles
- **Temporal Consistency** maintains recommendation stability

## Interactive Demo Features

### 1. User Recommendation Engine
- Input any user ID (1-500)
- Choose recommendation algorithm
- Get top-10 personalized recommendations
- View recommendation scores and movie details

### 2. Algorithm Comparison
- Compare all 5 methods side-by-side
- Analyze recommendation overlap
- Evaluate method-specific strengths

### 3. User Profile Explorer
- View complete user demographics
- Analyze rating history and patterns
- Understand user preference evolution

### 4. Movie Information System
- Detailed movie metadata
- Rating statistics and distributions
- Popularity and revenue analysis

## Research Applications

### Recommendation System Studies
- **Algorithm Comparison** across different user types
- **Cold Start Problem** analysis and solutions
- **Diversity vs. Accuracy** trade-off studies
- **Hybrid Method Optimization** weight tuning

### Machine Learning Education
- **Collaborative Filtering** implementation from scratch
- **Matrix Factorization** concepts and applications
- **Feature Engineering** for content-based systems
- **Evaluation Metrics** understanding and calculation

## Limitations & Future Work

### Current Limitations
- Synthetic data may not capture all real-world patterns
- No temporal dynamics in user preferences
- Limited demographic factors in user modeling
- Static movie catalog without new releases

### Planned Enhancements
- [ ] **Real Movie Database** integration (TMDB/IMDb APIs)
- [ ] **Deep Learning Models** (Neural Collaborative Filtering)
- [ ] **Sequential Recommendations** with RNNs/Transformers
- [ ] **Multi-Modal Content** (posters, trailers, reviews)
- [ ] **Explainable Recommendations** with reasoning
- [ ] **A/B Testing Framework** for algorithm evaluation

### Advanced Algorithms
- [ ] **Autoencoders** for dimensionality reduction
- [ ] **Graph Neural Networks** for user-item interactions
- [ ] **Reinforcement Learning** for long-term user satisfaction
- [ ] **Contextual Bandits** for real-time personalization

## Dataset Schema

### Ratings Table
| Column | Type | Description |
|--------|------|-------------|
| user_id | int | Unique user identifier |
| movie_id | int | Unique movie identifier |
| rating | float | Rating (1.0-5.0, 0.5 increments) |
| timestamp | int | Unix timestamp |

### Movies Table
| Column | Type | Description |
|--------|------|-------------|
| movie_id | int | Unique movie identifier |
| title | string | Movie title |
| genres | string | Pipe-separated genres |
| year | int | Release year |
| runtime | int | Duration in minutes |
| director | string | Director name |
| cast | string | Pipe-separated cast |
| budget | float | Production budget |
| revenue | float | Box office revenue |

### Users Table
| Column | Type | Description |
|--------|------|-------------|
| user_id | int | Unique user identifier |
| age_group | string | Age category |
| gender | string | M/F |
| occupation | string | Job category |
| favorite_genres | string | Pipe-separated preferences |

## Contributing

This project is part of the **tihassfjord ML Portfolio**. Contributions welcome:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## License

MIT License - See LICENSE file for details

## References

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems
- Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook
- Aggarwal, C. C. (2016). Recommender Systems: The Textbook

## Contact

**GitHub:** [tihassfjord](https://github.com/tihassfjord)  
**Project:** Movie Recommendation System with Multiple Algorithms

---

*Part of the Machine Learning Project Portfolio by tihassfjord*
