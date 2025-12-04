import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class NewsFeatureExtractor:
    """
    Extract enhanced linguistically-motivated features for news classification.
    Optimized for Maximum Entropy with numeric and combined features.
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        # Morphological patterns
        self.suffix_patterns = ['-tion', '-ness', '-ly', '-ment', '-ing', '-ed', '-ful', '-ous', '-ive', '-able']
        self.prefix_patterns = ['un-', 're-', 'dis-', 'non-', 'over-', 'pre-']
        
        # News-specific domain terminology (expanded)
        self.politics_words = {'election', 'vote', 'president', 'congress', 'senate', 'bill', 
                              'policy', 'government', 'political', 'campaign', 'democrat', 'republican',
                              'legislature', 'law', 'administration', 'governor', 'mayor'}
        
        self.business_words = {'market', 'stock', 'company', 'business', 'economy', 'trade',
                              'revenue', 'profit', 'investment', 'financial', 'corporate', 'ceo',
                              'earnings', 'sales', 'consumer', 'retail', 'industry', 'quarter'}
        
        self.tech_words = {'technology', 'software', 'app', 'digital', 'computer', 'internet',
                          'tech', 'data', 'online', 'cyber', 'ai', 'device', 'startup',
                          'platform', 'innovation', 'smartphone', 'google', 'apple', 'facebook'}
        
        self.sports_words = {'game', 'team', 'player', 'win', 'score', 'league', 'season',
                            'coach', 'championship', 'tournament', 'match', 'athlete', 'football',
                            'basketball', 'baseball', 'soccer', 'nfl', 'nba'}
        
        self.entertainment_words = {'movie', 'film', 'show', 'music', 'star', 'celebrity',
                                   'actor', 'director', 'album', 'concert', 'performance',
                                   'hollywood', 'television', 'series', 'drama', 'comedy'}
        
        self.world_news_words = {'country', 'international', 'foreign', 'nation', 'global',
                                'minister', 'embassy', 'treaty', 'diplomacy', 'border',
                                'war', 'conflict', 'peace', 'alliance', 'united nations'}
        
        self.crime_words = {'police', 'arrest', 'crime', 'murder', 'shooting', 'killed',
                           'charged', 'suspect', 'investigation', 'victim', 'court',
                           'trial', 'jury', 'sentence', 'convicted', 'robbery'}
        
        self.health_words = {'health', 'medical', 'doctor', 'hospital', 'patient', 'disease',
                            'treatment', 'vaccine', 'drug', 'medicine', 'study', 'research'}
        
        self.science_words = {'science', 'research', 'study', 'scientist', 'discovery',
                             'experiment', 'theory', 'analysis', 'data', 'evidence'}
        
        self.environment_words = {'climate', 'environment', 'energy', 'pollution', 'green',
                                 'carbon', 'renewable', 'fossil', 'sustainability', 'earth'}
        
        # Sentiment/tone indicators
        self.urgent_words = {'breaking', 'urgent', 'alert', 'emergency', 'crisis', 'critical'}
        self.positive_words = {'win', 'success', 'celebrate', 'achievement', 'victory', 'amazing',
                              'great', 'excellent', 'wonderful', 'love', 'triumph', 'gain'}
        self.negative_words = {'death', 'killed', 'disaster', 'tragedy', 'failure', 'crisis',
                              'scandal', 'controversy', 'attack', 'violence', 'lose', 'decline'}
        
        # News-specific patterns
        self.question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which'}
        
        # Passive voice indicators
        self.passive_auxiliaries = {'is', 'are', 'was', 'were', 'been', 'be'}
        
    def extract_features(self, text):
        """Extract enhanced features for Maximum Entropy classifier."""
        features = {}
        
        if pd.isna(text) or not isinstance(text, str):
            return features
        
        text_lower = text.lower()
        
        # Tokenization
        words = word_tokenize(text_lower)
        words_clean = [w for w in words if w.isalpha()]
        
        if len(words_clean) == 0:
            return features
        
        # POS tagging
        pos_tags = pos_tag(words)
        
        # === 1. WORD FEATURES ===
        # Word frequencies (numeric - MaxEnt handles these well)
        word_freq = Counter(words_clean)
        for word, freq in word_freq.most_common(50):  # Top 50 words
            features[f'word_{word}'] = min(freq, 3)  # Cap at 3
        
        # === 2. BIGRAM FEATURES ===
        bigrams = [' '.join([words_clean[i], words_clean[i+1]]) 
                   for i in range(len(words_clean)-1)]
        bigram_freq = Counter(bigrams)
        for bigram, freq in bigram_freq.most_common(15):  # Top 15 bigrams
            features[f'bigram_{bigram}'] = min(freq, 2)
        
        # === 3. MORPHOLOGICAL FEATURES ===
        # Suffix counts
        for suffix in self.suffix_patterns:
            suffix_clean = suffix.replace('-', '')
            count = sum(1 for word in words_clean if word.endswith(suffix_clean))
            if count > 0:
                features[f'suffix{suffix}_count'] = min(count, 3)
        
        # Prefix counts
        for prefix in self.prefix_patterns:
            prefix_clean = prefix.replace('-', '')
            count = sum(1 for word in words_clean if word.startswith(prefix_clean))
            if count > 0:
                features[f'prefix{prefix}_count'] = min(count, 3)
        
        # === 4. SYNTACTIC FEATURES ===
        # POS tag counts
        pos_counts = Counter([tag[:2] for _, tag in pos_tags])
        for pos, count in pos_counts.items():
            features[f'pos_{pos}_count'] = min(count, 10)
        
        # POS ratios (useful for MaxEnt)
        total_words = len(pos_tags)
        features['noun_ratio'] = pos_counts.get('NN', 0) / total_words
        features['verb_ratio'] = pos_counts.get('VB', 0) / total_words
        features['adj_ratio'] = pos_counts.get('JJ', 0) / total_words
        features['proper_noun_ratio'] = pos_counts.get('NN', 0) / total_words  # NNP
        
        # Named entity indicators (proper nouns very important in news)
        proper_noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NNP'))
        if proper_noun_count > 0:
            features['proper_noun_count'] = min(proper_noun_count, 10)
        
        # Numbers/dates (critical in news)
        number_count = sum(1 for _, tag in pos_tags if tag == 'CD')
        if number_count > 0:
            features['number_count'] = min(number_count, 5)
        
        # Passive voice count (common in news reporting)
        passive_count = 0
        for i in range(len(pos_tags) - 1):
            word, tag = pos_tags[i]
            next_tag = pos_tags[i + 1][1]
            if word.lower() in self.passive_auxiliaries and next_tag == 'VBN':
                passive_count += 1
        if passive_count > 0:
            features['passive_voice_count'] = min(passive_count, 3)
        
        # Question indicators (titles often have questions)
        if any(word in self.question_words for word in words_clean[:5]):  # First 5 words
            features['starts_with_question'] = 1
        
        # === 5. SEMANTIC FEATURES ===
        # Lemmatize for better matching
        lemmatized_words = set()
        for word in words_clean:
            lemma_v = self.lemmatizer.lemmatize(word, pos='v')
            lemmatized_words.add(lemma_v)
            lemma_n = self.lemmatizer.lemmatize(word, pos='n')
            lemmatized_words.add(lemma_n)
        
        # Domain-specific term counts (numeric features)
        features['politics_term_count'] = len(lemmatized_words & self.politics_words)
        features['business_term_count'] = len(lemmatized_words & self.business_words)
        features['tech_term_count'] = len(lemmatized_words & self.tech_words)
        features['sports_term_count'] = len(lemmatized_words & self.sports_words)
        features['entertainment_term_count'] = len(lemmatized_words & self.entertainment_words)
        features['world_news_term_count'] = len(lemmatized_words & self.world_news_words)
        features['crime_term_count'] = len(lemmatized_words & self.crime_words)
        features['health_term_count'] = len(lemmatized_words & self.health_words)
        features['science_term_count'] = len(lemmatized_words & self.science_words)
        features['environment_term_count'] = len(lemmatized_words & self.environment_words)
        
        # Tone indicators (numeric)
        features['urgent_term_count'] = len(lemmatized_words & self.urgent_words)
        features['positive_term_count'] = len(lemmatized_words & self.positive_words)
        features['negative_term_count'] = len(lemmatized_words & self.negative_words)
        
        # Sentiment ratio
        pos_count = features['positive_term_count']
        neg_count = features['negative_term_count']
        if pos_count + neg_count > 0:
            features['sentiment_ratio'] = (pos_count - neg_count) / (pos_count + neg_count)
        
        # === 6. STRUCTURAL FEATURES ===
        features['text_length'] = min(len(text), 500) / 500  # Normalized
        features['word_count'] = min(len(words_clean), 100) / 100  # Normalized
        
        if len(words_clean) > 0:
            features['avg_word_length'] = sum(len(w) for w in words_clean) / len(words_clean) / 10
        
        # Punctuation features (important in news)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['quote_count'] = text.count('"') + text.count("'")
        
        # Capitalization patterns
        features['all_caps_words'] = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
        
        # === 7. FEATURE COMBINATIONS (MaxEnt strength) ===
        # Domain + sentiment combinations
        if features.get('politics_term_count', 0) > 0:
            if features.get('negative_term_count', 0) > 0:
                features['politics_negative_combo'] = 1
            if features.get('urgent_term_count', 0) > 0:
                features['politics_urgent_combo'] = 1
        
        if features.get('business_term_count', 0) > 0:
            if features.get('positive_term_count', 0) > 0:
                features['business_positive_combo'] = 1
            if features.get('number_count', 0) > 0:
                features['business_numbers_combo'] = 1
        
        if features.get('sports_term_count', 0) > 0:
            if features.get('positive_term_count', 0) > 0:
                features['sports_positive_combo'] = 1
            if features.get('number_count', 0) > 0:
                features['sports_numbers_combo'] = 1
        
        if features.get('crime_term_count', 0) > 0:
            if features.get('negative_term_count', 0) > 0:
                features['crime_negative_combo'] = 1
        
        if features.get('tech_term_count', 0) > 0:
            if features.get('business_term_count', 0) > 0:
                features['tech_business_combo'] = 1
        
        # Proper nouns + domain combinations
        if features.get('proper_noun_count', 0) > 2:
            if features.get('politics_term_count', 0) > 0:
                features['proper_noun_politics_combo'] = 1
            if features.get('sports_term_count', 0) > 0:
                features['proper_noun_sports_combo'] = 1
            if features.get('entertainment_term_count', 0) > 0:
                features['proper_noun_entertainment_combo'] = 1
        
        return features


class MaxEntNewsClassifier:
    """
    Maximum Entropy classifier for news classification using Logistic Regression.
    Optimized for news articles with many categories.
    """
    
    def __init__(self, C=1.0, max_iter=1000, top_features=5000):
        """
        Initialize MaxEnt classifier.
        
        Args:
            C: Inverse regularization strength (smaller = stronger regularization)
            max_iter: Maximum iterations for optimization
            top_features: Maximum number of features to use
        """
        self.C = C
        self.max_iter = max_iter
        self.top_features = top_features
        self.model = None
        self.feature_extractor = NewsFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def _features_to_vector(self, features_dict):
        """Convert feature dictionary to vector."""
        vector = np.zeros(len(self.feature_names))
        for i, feature_name in enumerate(self.feature_names):
            vector[i] = features_dict.get(feature_name, 0)
        return vector
        
    def fit(self, X, y):
        """Train the Maximum Entropy classifier."""
        print("Extracting features from training data...")
        n_samples = len(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Extract features with progress
        all_features = []
        print(f"Processing {n_samples} training documents...")
        for idx, text in enumerate(X):
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1}/{n_samples} documents ({100*(idx+1)/n_samples:.1f}%)")
            features = self.feature_extractor.extract_features(text)
            all_features.append(features)
        print(f"  Completed: {n_samples}/{n_samples} documents (100.0%)")
        
        # Build feature vocabulary with frequency-based selection
        print("\nBuilding feature vocabulary...")
        feature_counter = Counter()
        for features in all_features:
            feature_counter.update(features.keys())
        
        # Select top features by frequency
        if len(feature_counter) > self.top_features:
            self.feature_names = [f for f, _ in feature_counter.most_common(self.top_features)]
        else:
            self.feature_names = list(feature_counter.keys())
        
        print(f"Total unique features: {len(feature_counter)}")
        print(f"Using {len(self.feature_names)} features")
        
        # Convert to feature matrix
        print("\nConverting to feature matrix...")
        X_matrix = np.zeros((n_samples, len(self.feature_names)))
        for i, features in enumerate(all_features):
            X_matrix[i] = self._features_to_vector(features)
        
        print(f"Feature matrix shape: {X_matrix.shape}")
        
        # Train Logistic Regression (Maximum Entropy) model
        print("\nTraining Maximum Entropy model...")
        print("This may take a while for large datasets with many categories...")
        
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver='lbfgs',  # Good for multiclass
            multi_class='multinomial',  # True MaxEnt
            random_state=42,
            verbose=1,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.model.fit(X_matrix, y_encoded)
        print("Training complete!")
        
    def predict(self, X):
        """Predict class labels."""
        print(f"\nMaking predictions on {len(X)} test documents...")
        
        # Extract features
        all_features = []
        for idx, text in enumerate(X):
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1}/{len(X)} documents ({100*(idx+1)/len(X):.1f}%)")
            features = self.feature_extractor.extract_features(text)
            all_features.append(features)
        print(f"  Completed: {len(X)}/{len(X)} documents (100.0%)")
        
        # Convert to matrix
        X_matrix = np.zeros((len(X), len(self.feature_names)))
        for i, features in enumerate(all_features):
            X_matrix[i] = self._features_to_vector(features)
        
        # Predict
        y_pred_encoded = self.model.predict(X_matrix)
        predictions = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return predictions
    
    def get_top_features(self, n=20):
        """Get top features for each class."""
        if self.model is None:
            return None
        
        top_features = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            coefficients = self.model.coef_[i]
            top_indices = np.argsort(np.abs(coefficients))[-n:][::-1]
            top_features[class_name] = [
                (self.feature_names[idx], coefficients[idx]) 
                for idx in top_indices
            ]
        
        return top_features


def load_news_data(filepath='../data/News_Category_Dataset_v3.json'):
    """Load news dataset from JSON file."""
    print(f"Loading data from {filepath}...")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                article = json.loads(line)
                data.append(article)
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(data)
    
    # Use 'short_description' as the text to classify
    if 'short_description' not in df.columns or 'category' not in df.columns:
        raise ValueError("Dataset must have 'short_description' and 'category' columns")
    
    # Clean data
    df = df[['short_description', 'category']].dropna()
    
    print(f"\nTotal samples: {len(df)}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    print(f"\nNumber of unique categories: {df['category'].nunique()}")
    
    return df


def plot_confusion_matrix(y_true, y_pred, classes, save_path='../results/news/maxent_confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # For many categories, use smaller font and larger figure
    n_classes = len(classes)
    figsize = (max(14, n_classes * 0.7), max(12, n_classes * 0.6))
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=classes, yticklabels=classes, 
                cbar_kws={'label': 'Count'}, annot_kws={'size': 7})
    plt.title('Confusion Matrix - MaxEnt News Classification', fontsize=14, pad=15)
    plt.ylabel('True Category', fontsize=11)
    plt.xlabel('Predicted Category', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def evaluate_classifier(y_true, y_pred, classes, save_path='../results/news/maxent_evaluation_metrics.txt'):
    """Calculate and display evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    # Prepare output text
    output_lines = []
    output_lines.append("="*80)
    output_lines.append("MAXIMUM ENTROPY CLASSIFIER - EVALUATION METRICS")
    output_lines.append("="*80)
    output_lines.append(f"\nOverall Accuracy: {accuracy:.4f}\n")
    output_lines.append("-"*80)
    output_lines.append("PER-CLASS METRICS")
    output_lines.append("-"*80)
    output_lines.append(f"{'Category':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    output_lines.append("-"*80)
    
    for i, cls in enumerate(classes):
        output_lines.append(f"{cls:<25} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10.0f}")
    
    output_lines.append("-"*80)
    output_lines.append(f"{'Macro Average':<25} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}")
    output_lines.append(f"{'Micro Average':<25} {precision_micro:<12.4f} {recall_micro:<12.4f} {f1_micro:<12.4f}")
    output_lines.append("="*80)
    
    # Print to console
    print("\n" + "\n".join(output_lines))
    
    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("\n".join(output_lines))
    print(f"\nEvaluation metrics saved to: {save_path}")
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }


def main():
    """Main pipeline."""
    print("="*90)
    print("MAXIMUM ENTROPY NEWS CATEGORY CLASSIFICATION")
    print("WITH ENHANCED LINGUISTICALLY-MOTIVATED FEATURES")
    print("="*90)
    
    print("\n1. Loading data...")
    df = load_news_data('../data/News_Category_Dataset_v3.json')
    
    print("\n2. Splitting data (70% train, 30% test)...")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df_shuffled['short_description'].tolist()
    y = df_shuffled['category'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    print("\n3. Training Maximum Entropy classifier...")
    # C=1.0 provides moderate regularization
    # Increase C for less regularization, decrease for more
    classifier = MaxEntNewsClassifier(C=1.0, max_iter=1000, top_features=5000)
    classifier.fit(X_train, y_train)
    
    print(f"\nNumber of classes: {len(set(y_train))}")
    
    print("\n4. Making predictions...")
    y_pred = classifier.predict(X_test)
    
    print("\n5. Evaluating performance...")
    metrics = evaluate_classifier(y_test, y_pred, sorted(set(y_test)))
    
    print("\n6. Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, sorted(set(y_test)))
    
    print("\n7. Top features per category (sample):")
    top_features = classifier.get_top_features(n=10)
    sample_categories = list(top_features.keys())[:5]  # Show first 5 categories
    for category in sample_categories:
        print(f"\n{category}:")
        for feature, weight in top_features[category][:5]:
            print(f"  {feature}: {weight:.4f}")
    
    print("\n" + "="*90)
    print("CLASSIFICATION COMPLETE!")
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Macro F1: {metrics['f1_macro']:.4f}")
    print("="*90)


if __name__ == "__main__":
    main()
