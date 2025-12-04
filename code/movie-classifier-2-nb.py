import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

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


class FeatureExtractor:
    """
    Extract linguistically-motivated features for text classification.
    Optimized for faster computation while maintaining linguistic richness.
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        # Key morphological patterns (reduced set)
        self.suffix_patterns = ['-tion', '-ness', '-ly', '-ment', '-ing', '-ed', '-ful', '-ous', '-ive', '-able']
        self.prefix_patterns = ['un-', 're-', 'dis-', 'non-', 'over-']
        
        # Sentiment indicators
        self.positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                               'love', 'best', 'beautiful', 'perfect', 'happy', 'brilliant'}
        self.negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor',
                               'sad', 'angry', 'evil', 'dark', 'death', 'fear'}
        
        # Domain-specific terminology
        self.action_words = {'fight', 'battle', 'war', 'mission', 'escape', 'chase', 'explosion',
                            'weapon', 'soldier', 'attack'}
        self.romance_words = {'love', 'relationship', 'romance', 'marry', 'wedding', 'heart',
                             'kiss', 'couple', 'passionate'}
        self.horror_words = {'horror', 'ghost', 'haunted', 'dead', 'death', 'blood', 'murder',
                            'monster', 'terror', 'evil'}
        self.comedy_words = {'comedy', 'funny', 'laugh', 'humor', 'hilarious', 'joke'}
        self.scifi_words = {'space', 'future', 'alien', 'robot', 'technology', 'planet',
                           'time', 'science', 'virtual'}
        
        # Passive voice indicators
        self.passive_auxiliaries = {'is', 'are', 'was', 'were', 'been', 'be'}
        
    def extract_features(self, text):
        """Extract optimized linguistic features."""
        features = {}
        
        if pd.isna(text):
            return features
        
        text_lower = text.lower()
        
        # Tokenization
        sentences = sent_tokenize(text)
        words = word_tokenize(text_lower)
        words_clean = [w for w in words if w.isalpha()]
        
        # POS tagging (only once)
        pos_tags = pos_tag(words)
        
        # === 1. WORD FEATURES (Most discriminative) ===
        # Only keep words, skip character n-grams for speed
        for word in words_clean:
            features[f'word_{word}'] = 1
        
        # === 2. MORPHOLOGICAL FEATURES ===
        # Suffix patterns
        for suffix in self.suffix_patterns:
            suffix_clean = suffix.replace('-', '')
            if any(word.endswith(suffix_clean) for word in words_clean):
                features[f'has_suffix{suffix}'] = 1
        
        # Prefix patterns
        for prefix in self.prefix_patterns:
            prefix_clean = prefix.replace('-', '')
            if any(word.startswith(prefix_clean) for word in words_clean):
                features[f'has_prefix{prefix}'] = 1
        
        # === 3. SYNTACTIC FEATURES ===
        # POS tag presence (major categories only)
        pos_set = set([tag for _, tag in pos_tags])
        major_pos = ['NN', 'VB', 'JJ', 'RB', 'PRP', 'IN', 'DT']
        for tag_prefix in major_pos:
            if any(tag.startswith(tag_prefix) for tag in pos_set):
                features[f'has_pos_{tag_prefix}'] = 1
        
        # Named entity indicators (simple)
        words_original = word_tokenize(text)
        has_capitalized = any(w[0].isupper() for i, w in enumerate(words_original) if i > 0 and w.isalpha())
        if has_capitalized:
            features['has_named_entity'] = 1
        
        if re.search(r'\b(19|20)\d{2}\b', text):
            features['has_date_year'] = 1
        
        # Passive voice
        for i in range(len(pos_tags) - 1):
            word, tag = pos_tags[i]
            next_tag = pos_tags[i + 1][1]
            if word.lower() in self.passive_auxiliaries and next_tag == 'VBN':
                features['has_passive_voice'] = 1
                break
        
        # === 4. SEMANTIC FEATURES ===
        # Lemmatize words for better domain/sentiment matching
        lemmatized_words = set()
        for word in words_clean:
            # Try verb lemmatization first (most important for actions)
            lemma_v = self.lemmatizer.lemmatize(word, pos='v')
            lemmatized_words.add(lemma_v)
            # Also try noun lemmatization
            lemma_n = self.lemmatizer.lemmatize(word, pos='n')
            lemmatized_words.add(lemma_n)
        
        # Domain-specific terminology (checked against lemmas)
        if lemmatized_words & self.action_words:
            features['has_action_terms'] = 1
        if lemmatized_words & self.romance_words:
            features['has_romance_terms'] = 1
        if lemmatized_words & self.horror_words:
            features['has_horror_terms'] = 1
        if lemmatized_words & self.comedy_words:
            features['has_comedy_terms'] = 1
        if lemmatized_words & self.scifi_words:
            features['has_scifi_terms'] = 1
        
        # Sentiment (also checked against lemmas)
        if lemmatized_words & self.positive_words:
            features['has_positive_sentiment'] = 1
        if lemmatized_words & self.negative_words:
            features['has_negative_sentiment'] = 1
        
        return features


class NaiveBayesTextClassifier:
    """
    Naive Bayes classifier optimized for speed while maintaining linguistic features.
    """
    
    def __init__(self, alpha=1.0, top_features=3000):
        self.alpha = alpha
        self.top_features = top_features
        self.class_priors = {}
        self.feature_probs = {}
        self.feature_set = set()
        self.classes = []
        self.feature_extractor = FeatureExtractor()
        
    def fit(self, X, y):
        """Train the classifier."""
        print("Extracting features from training data...")
        self.classes = list(set(y))
        n_samples = len(y)
        
        # Class priors
        class_counts = Counter(y)
        for cls in self.classes:
            self.class_priors[cls] = class_counts[cls] / n_samples
        
        # Extract features with progress
        all_features = []
        print(f"Processing {n_samples} training documents...")
        for idx, text in enumerate(X):
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{n_samples} documents ({100*(idx+1)/n_samples:.1f}%)")
            features = self.feature_extractor.extract_features(text)
            all_features.append(features)
        print(f"  Completed: {n_samples}/{n_samples} documents (100.0%)")
        
        # Build feature vocabulary
        feature_counter = Counter()
        for features in all_features:
            feature_counter.update(features.keys())
        
        # Keep top features
        if len(feature_counter) > self.top_features:
            self.feature_set = set([f for f, _ in feature_counter.most_common(self.top_features)])
        else:
            self.feature_set = set(feature_counter.keys())
        
        print(f"Total unique features: {len(feature_counter)}")
        print(f"Using top {len(self.feature_set)} features")
        
        # Calculate feature probabilities
        class_feature_counts = {cls: defaultdict(int) for cls in self.classes}
        class_doc_counts = {cls: 0 for cls in self.classes}
        
        for features, label in zip(all_features, y):
            class_doc_counts[label] += 1
            for feature in self.feature_set:
                if features.get(feature, 0) == 1:
                    class_feature_counts[label][feature] += 1
        
        # P(feature|class) with Laplace smoothing
        for cls in self.classes:
            self.feature_probs[cls] = {}
            total_docs = class_doc_counts[cls]
            
            for feature in self.feature_set:
                count = class_feature_counts[cls][feature]
                self.feature_probs[cls][feature] = (count + self.alpha) / (total_docs + 2 * self.alpha)
    
    def predict(self, X):
        """Predict class labels."""
        print(f"Making predictions on {len(X)} test documents...")
        predictions = []
        for idx, text in enumerate(X):
            if (idx + 1) % 500 == 0:
                print(f"  Predicted {idx + 1}/{len(X)} documents ({100*(idx+1)/len(X):.1f}%)")
            scores = self.predict_proba_single(text)
            predictions.append(max(scores, key=scores.get))
        print(f"  Completed: {len(X)}/{len(X)} documents (100.0%)")
        return predictions
    
    def predict_proba_single(self, text):
        """Predict probabilities for single document."""
        features = self.feature_extractor.extract_features(text)
        class_scores = {}
        
        for cls in self.classes:
            log_prob = np.log(self.class_priors[cls])
            
            for feature in self.feature_set:
                if features.get(feature, 0) == 1:
                    log_prob += np.log(self.feature_probs[cls][feature])
                else:
                    log_prob += np.log(1 - self.feature_probs[cls][feature])
            
            class_scores[cls] = log_prob
        
        return class_scores


def load_data(filepath='../data/movies_processed.csv'):
    """Load the preprocessed movie data from CSV."""
    print(f"Loading data from: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} samples")
        
        # Check required columns
        if 'summary' not in df.columns or 'genres' not in df.columns:
            raise ValueError("CSV must contain 'summary' and 'genres' columns")
        
        # For multi-label movies, take only the first genre
        df['genre'] = df['genres'].apply(lambda x: x.split('|')[0] if pd.notna(x) else None)
        
        # Remove rows with missing summaries or genres
        df = df.dropna(subset=['summary', 'genre'])
        
        print(f"\nGenre distribution:")
        print(df['genre'].value_counts())
        
        # NO FILTERING - Keep all genres regardless of frequency
        print(f"\nTotal samples: {len(df)}")
        print(f"Number of genres: {df['genre'].nunique()}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        print("Please ensure movies_processed.csv exists in the data directory")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def plot_confusion_matrix(y_true, y_pred, classes, save_path='../results/CMUmovies/confusion_matrix_naivebayes.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - NB with Linguistic Features (CMU Movies)', fontsize=16, pad=20)
    plt.ylabel('True Genre', fontsize=12)
    plt.xlabel('Predicted Genre', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def evaluate_classifier(y_true, y_pred, classes, save_path='../results/CMUmovies/naivebayes_evaluation_metrics.txt'):
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
    output_lines.append("EVALUATION METRICS")
    output_lines.append("="*80)
    output_lines.append(f"\nOverall Accuracy: {accuracy:.4f}\n")
    output_lines.append("-"*80)
    output_lines.append("PER-CLASS METRICS")
    output_lines.append("-"*80)
    output_lines.append(f"{'Genre':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    output_lines.append("-"*80)
    
    for i, cls in enumerate(classes):
        output_lines.append(f"{cls:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10.0f}")
    
    output_lines.append("-"*80)
    output_lines.append(f"{'Macro Average':<20} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}")
    output_lines.append(f"{'Micro Average':<20} {precision_micro:<12.4f} {recall_micro:<12.4f} {f1_micro:<12.4f}")
    output_lines.append("="*80)
    
    # Print to console
    print("\n" + "\n".join(output_lines))
    
    # Save to file
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
    print("="*80)
    print("NAIVE BAYES MOVIE GENRE CLASSIFICATION - CMU DATASET")
    print("WITH LINGUISTICALLY-MOTIVATED FEATURES")
    print("="*80)
    
    print("\n1. Loading data...")
    df = load_data('../data/movies_processed.csv')
    
    print("\n2. Splitting data (70% train, 30% test)...")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df_shuffled['summary'].tolist()
    y = df_shuffled['genre'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    print("\n3. Training classifier...")
    classifier = NaiveBayesTextClassifier(alpha=1.0, top_features=3000)
    classifier.fit(X_train, y_train)
    
    print("\n4. Making predictions...")
    y_pred = classifier.predict(X_test)
    
    print("\n5. Evaluating performance...")
    metrics = evaluate_classifier(y_test, y_pred, sorted(classifier.classes))
    
    print("\n6. Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, sorted(classifier.classes))
    
    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()