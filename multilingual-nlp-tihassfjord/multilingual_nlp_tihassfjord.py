#!/usr/bin/env python3
"""
Multi-language NLP Pipeline (highlight) — tihassfjord

Advanced natural language processing pipeline supporting multiple languages
with translation, sentiment analysis, and named entity recognition using 
transformer models and language detection.

Author: tihassfjord
Project: Advanced ML Portfolio - Multilingual NLP
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np
from pathlib import Path

# NLP and ML libraries
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, pipeline, AutoConfig
    )
    import torch
    from langdetect import detect, DetectorFactory
    from googletrans import Translator
    import spacy
except ImportError as e:
    print(f"Missing required libraries. Install with: pip install -r requirements.txt")
    print(f"Error: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multilingual_nlp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 42

class MultilingualNLPPipeline:
    """
    Advanced multilingual NLP pipeline with translation, sentiment analysis,
    and named entity recognition capabilities.
    """
    
    def __init__(self):
        """Initialize the multilingual NLP pipeline."""
        self.translator = Translator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"tihassfjord: Initializing multilingual NLP pipeline on {self.device}")
        
        # Initialize models
        self.models = self._initialize_models()
        self.supported_languages = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese',
            'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi'
        }
        
        logger.info("tihassfjord: Multilingual NLP pipeline initialized successfully")
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize various NLP models."""
        models = {}
        
        try:
            # Sentiment analysis model (multilingual)
            logger.info("Loading multilingual sentiment model...")
            models['sentiment'] = pipeline(
                'sentiment-analysis',
                model='nlptown/bert-base-multilingual-uncased-sentiment',
                device=0 if self.device.type == 'cuda' else -1,
                return_all_scores=True
            )
            
            # Named Entity Recognition model
            logger.info("Loading multilingual NER model...")
            models['ner'] = pipeline(
                'ner',
                model='xlm-roberta-base-ner',
                aggregation_strategy='simple',
                device=0 if self.device.type == 'cuda' else -1
            )
            
            # Text classification for language-specific tasks
            logger.info("Loading text classification model...")
            models['classification'] = pipeline(
                'text-classification',
                model='facebook/bart-large-mnli',
                device=0 if self.device.type == 'cuda' else -1
            )
            
        except Exception as e:
            logger.warning(f"Some models failed to load: {e}")
            # Fallback to simpler models
            models['sentiment'] = pipeline('sentiment-analysis', device=-1)
            models['ner'] = pipeline('ner', aggregation_strategy='simple', device=-1)
            
        return models
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of input text.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            if len(text.strip()) < 3:
                return 'unknown', 0.0
                
            lang_code = detect(text)
            confidence = 0.95  # langdetect doesn't provide confidence, estimate high
            
            return lang_code, confidence
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'unknown', 0.0
    
    def translate_text(self, text: str, target_lang: str = 'en', source_lang: str = None) -> Dict[str, Any]:
        """
        Translate text to target language.
        
        Args:
            text: Input text to translate
            target_lang: Target language code (default: 'en')
            source_lang: Source language code (auto-detect if None)
            
        Returns:
            Dictionary with translation results
        """
        try:
            if source_lang is None:
                detected_lang, confidence = self.detect_language(text)
                source_lang = detected_lang
            
            if source_lang == target_lang:
                return {
                    'original_text': text,
                    'translated_text': text,
                    'source_language': source_lang,
                    'target_language': target_lang,
                    'translation_confidence': 1.0
                }
            
            # Perform translation
            result = self.translator.translate(text, src=source_lang, dest=target_lang)
            
            return {
                'original_text': text,
                'translated_text': result.text,
                'source_language': source_lang,
                'target_language': target_lang,
                'translation_confidence': getattr(result, 'confidence', 0.8)
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': 'unknown',
                'target_language': target_lang,
                'translation_confidence': 0.0,
                'error': str(e)
            }
    
    def analyze_sentiment(self, text: str, language: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment of text in multiple languages.
        
        Args:
            text: Input text for sentiment analysis
            language: Language code (auto-detect if None)
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            if language is None:
                language, _ = self.detect_language(text)
            
            # Use multilingual sentiment model
            results = self.models['sentiment'](text)
            
            # Process results based on model output format
            if isinstance(results[0], list):
                # Model returns all scores
                sentiment_scores = {item['label']: item['score'] for item in results[0]}
                top_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
                sentiment_label = top_sentiment[0]
                confidence = top_sentiment[1]
            else:
                # Model returns single prediction
                sentiment_label = results[0]['label']
                confidence = results[0]['score']
                sentiment_scores = {sentiment_label: confidence}
            
            # Normalize sentiment labels
            sentiment_mapping = {
                'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral',
                'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive',
                '1 star': 'very_negative', '2 stars': 'negative', '3 stars': 'neutral',
                '4 stars': 'positive', '5 stars': 'very_positive'
            }
            
            normalized_sentiment = sentiment_mapping.get(sentiment_label.upper(), sentiment_label.lower())
            
            return {
                'text': text,
                'language': language,
                'sentiment': normalized_sentiment,
                'confidence': confidence,
                'all_scores': sentiment_scores,
                'model_used': 'multilingual-bert-sentiment'
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                'text': text,
                'language': language or 'unknown',
                'sentiment': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def extract_entities(self, text: str, language: str = None) -> Dict[str, Any]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text for entity extraction
            language: Language code (auto-detect if None)
            
        Returns:
            Dictionary with named entities
        """
        try:
            if language is None:
                language, _ = self.detect_language(text)
            
            # Use multilingual NER model
            entities = self.models['ner'](text)
            
            # Group entities by type
            entity_groups = {}
            for entity in entities:
                entity_type = entity['entity_group']
                if entity_type not in entity_groups:
                    entity_groups[entity_type] = []
                
                entity_groups[entity_type].append({
                    'text': entity['word'],
                    'confidence': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                })
            
            return {
                'text': text,
                'language': language,
                'entities': entities,
                'entity_groups': entity_groups,
                'entity_count': len(entities),
                'model_used': 'xlm-roberta-ner'
            }
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {
                'text': text,
                'language': language or 'unknown',
                'entities': [],
                'entity_groups': {},
                'entity_count': 0,
                'error': str(e)
            }
    
    def analyze_text_classification(self, text: str, categories: List[str], language: str = None) -> Dict[str, Any]:
        """
        Classify text into custom categories using zero-shot classification.
        
        Args:
            text: Input text to classify
            categories: List of possible categories
            language: Language code (auto-detect if None)
            
        Returns:
            Dictionary with classification results
        """
        try:
            if language is None:
                language, _ = self.detect_language(text)
            
            # Use zero-shot classification
            classifier = pipeline(
                'zero-shot-classification',
                model='facebook/bart-large-mnli',
                device=0 if self.device.type == 'cuda' else -1
            )
            
            results = classifier(text, categories)
            
            return {
                'text': text,
                'language': language,
                'predicted_category': results['labels'][0],
                'confidence': results['scores'][0],
                'all_categories': dict(zip(results['labels'], results['scores'])),
                'model_used': 'bart-large-mnli'
            }
            
        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            return {
                'text': text,
                'language': language or 'unknown',
                'predicted_category': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def process_text_comprehensive(self, text: str, target_lang: str = 'en', 
                                 custom_categories: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive text processing pipeline.
        
        Args:
            text: Input text to process
            target_lang: Target language for translation
            custom_categories: Custom categories for classification
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"tihassfjord: Processing text comprehensively...")
        
        # Language detection
        detected_lang, lang_confidence = self.detect_language(text)
        
        # Translation (if needed)
        translation_result = self.translate_text(text, target_lang, detected_lang)
        
        # Use translated text for English-based models, original for multilingual models
        analysis_text = translation_result['translated_text'] if detected_lang != 'en' else text
        
        # Sentiment analysis
        sentiment_result = self.analyze_sentiment(text, detected_lang)
        
        # Named entity recognition
        ner_result = self.extract_entities(text, detected_lang)
        
        # Custom classification (if categories provided)
        classification_result = None
        if custom_categories:
            classification_result = self.analyze_text_classification(
                analysis_text, custom_categories, detected_lang
            )
        
        # Compile comprehensive results
        comprehensive_result = {
            'input_text': text,
            'detected_language': {
                'code': detected_lang,
                'name': self.supported_languages.get(detected_lang, 'Unknown'),
                'confidence': lang_confidence
            },
            'translation': translation_result,
            'sentiment': sentiment_result,
            'entities': ner_result,
            'classification': classification_result,
            'processing_metadata': {
                'pipeline_version': '1.0.0',
                'device_used': str(self.device),
                'author': 'tihassfjord'
            }
        }
        
        return comprehensive_result
    
    def process_file(self, file_path: str, output_path: str = None, 
                     target_lang: str = 'en', custom_categories: List[str] = None) -> pd.DataFrame:
        """
        Process text file and save results.
        
        Args:
            file_path: Path to input text file
            output_path: Path to save results (optional)
            target_lang: Target language for translation
            custom_categories: Custom categories for classification
            
        Returns:
            DataFrame with processing results
        """
        logger.info(f"tihassfjord: Processing file: {file_path}")
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks for processing (handle large files)
            chunks = self._split_text_into_chunks(content, max_length=512)
            
            results = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 10:  # Skip very short chunks
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    result = self.process_text_comprehensive(chunk, target_lang, custom_categories)
                    result['chunk_id'] = i + 1
                    results.append(result)
            
            # Convert to DataFrame for easy analysis
            df = self._results_to_dataframe(results)
            
            # Save results if output path provided
            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Results saved to: {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            raise
    
    def _split_text_into_chunks(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into manageable chunks."""
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert processing results to DataFrame."""
        rows = []
        
        for result in results:
            row = {
                'chunk_id': result.get('chunk_id', 1),
                'text': result['input_text'][:100] + '...' if len(result['input_text']) > 100 else result['input_text'],
                'detected_language': result['detected_language']['code'],
                'language_confidence': result['detected_language']['confidence'],
                'translated_text': result['translation']['translated_text'][:100] + '...' if len(result['translation']['translated_text']) > 100 else result['translation']['translated_text'],
                'sentiment': result['sentiment']['sentiment'],
                'sentiment_confidence': result['sentiment']['confidence'],
                'entity_count': result['entities']['entity_count'],
                'main_entities': ', '.join([f"{ent['word']} ({ent['entity_group']})" for ent in result['entities']['entities'][:3]]),
            }
            
            if result['classification']:
                row['classification'] = result['classification']['predicted_category']
                row['classification_confidence'] = result['classification']['confidence']
            
            rows.append(row)
        
        return pd.DataFrame(rows)


def create_sample_data():
    """Create sample multilingual text data for testing."""
    sample_texts = {
        'english.txt': "Hello, how are you today? This is a wonderful day for machine learning and artificial intelligence research.",
        'spanish.txt': "Hola, ¿cómo estás hoy? Este es un día maravilloso para el aprendizaje automático y la investigación en inteligencia artificial.",
        'french.txt': "Bonjour, comment allez-vous aujourd'hui? C'est une merveilleuse journée pour l'apprentissage automatique et la recherche en intelligence artificielle.",
        'german.txt': "Hallo, wie geht es dir heute? Dies ist ein wunderbarer Tag für maschinelles Lernen und Forschung zur künstlichen Intelligenz.",
        'mixed.txt': "Hello! Bonjour! Hola! Today we explore multilingual NLP. Heute erforschen wir mehrsprachige NLP. Aujourd'hui, nous explorons le NLP multilingue."
    }
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    for filename, content in sample_texts.items():
        file_path = data_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    logger.info("Sample multilingual data created in data/ directory")


def demo_multilingual_nlp():
    """Demonstration of the multilingual NLP pipeline."""
    print("\n" + "="*70)
    print("🌍 MULTILINGUAL NLP PIPELINE DEMO by tihassfjord")
    print("="*70)
    
    # Initialize pipeline
    nlp_pipeline = MultilingualNLPPipeline()
    
    # Sample texts in different languages
    sample_texts = [
        ("Hello, I love this product! It's amazing and works perfectly.", "English"),
        ("Bonjour, j'adore ce produit! Il est incroyable et fonctionne parfaitement.", "French"),
        ("Hola, ¡me encanta este producto! Es increíble y funciona perfectamente.", "Spanish"),
        ("Hallo, ich liebe dieses Produkt! Es ist erstaunlich und funktioniert perfekt.", "German"),
        ("The company Apple Inc. was founded by Steve Jobs in California, United States.", "English"),
        ("La empresa Microsoft fue fundada por Bill Gates en Estados Unidos.", "Spanish")
    ]
    
    print("\n🔍 Processing sample texts:")
    print("-" * 50)
    
    for i, (text, expected_lang) in enumerate(sample_texts, 1):
        print(f"\n📝 Sample {i} ({expected_lang}):")
        print(f"Text: {text}")
        
        # Comprehensive processing
        result = nlp_pipeline.process_text_comprehensive(
            text, 
            target_lang='en',
            custom_categories=['technology', 'business', 'education', 'entertainment', 'sports']
        )
        
        # Display results
        print(f"🌐 Detected Language: {result['detected_language']['name']} ({result['detected_language']['code']}) - {result['detected_language']['confidence']:.2f}")
        
        if result['translation']['source_language'] != 'en':
            print(f"🔄 Translation: {result['translation']['translated_text']}")
        
        print(f"😊 Sentiment: {result['sentiment']['sentiment']} (confidence: {result['sentiment']['confidence']:.2f})")
        
        if result['entities']['entities']:
            entities_str = ", ".join([f"{ent['word']} ({ent['entity_group']})" for ent in result['entities']['entities'][:3]])
            print(f"🏷️  Entities: {entities_str}")
        
        if result['classification']:
            print(f"📂 Category: {result['classification']['predicted_category']} (confidence: {result['classification']['confidence']:.2f})")
        
        print("-" * 50)
    
    # File processing demo
    print("\n📄 File Processing Demo:")
    create_sample_data()
    
    # Process a sample file
    sample_file = 'data/mixed.txt'
    if os.path.exists(sample_file):
        print(f"Processing file: {sample_file}")
        df_results = nlp_pipeline.process_file(
            sample_file, 
            output_path='results/multilingual_analysis.csv',
            custom_categories=['technology', 'greeting', 'education']
        )
        print(f"✅ File processed successfully! Results shape: {df_results.shape}")
        print("\nSample results:")
        print(df_results[['detected_language', 'sentiment', 'entity_count']].head())
    
    print("\n🎉 Multilingual NLP Pipeline Demo Complete!")
    print("Author: tihassfjord | Advanced ML Portfolio")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run demo
    try:
        demo_multilingual_nlp()
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
