import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
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
    Extract linguistically-motivated features for Maximum Entropy classification.
    Enhanced with feature combinations and numeric features for better MaxEnt performance.
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        # Key morphological patterns
        self.suffix_patterns = ['-tion', '-ness', '-ly', '-ment', '-ing', '-ed', '-ful', '-ous', '-ive', '-able']
        self.prefix_patterns = ['un-', 're-', 'dis-', 'non-', 'over-']
        
        # Sentiment indicators
        self.positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                               'love', 'best', 'beautiful', 'perfect', 'happy', 'brilliant'}
        self.negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor',
                               'sad', 'angry', 'evil', 'dark', 'death', 'fear'}
        
        # Domain-specific terminology (expanded for better discrimination)
        self.action_words = {'fight', 'battle', 'war', 'mission', 'escape', 'chase', 'explosion',
                            'weapon', 'soldier', 'attack', 'combat', 'destroy', 'rescue'}
        self.romance_words = {'love', 'relationship', 'romance', 'marry', 'wedding', 'heart',
                             'kiss', 'couple', 'passionate', 'affair', 'lover'}
        self.horror_words = {'horror', 'ghost', 'haunted', 'dead', 'death', 'blood', 'murder',
                            'monster', 'terror', 'evil', 'scream', 'nightmare', 'zombie'}
        self.comedy_words = {'comedy', 'funny', 'laugh', 'humor', 'hilarious', 'joke', 'wit'}
        self.scifi_words = {'space', 'future', 'alien', 'robot', 'technology', 'planet',
                           'time', 'science', 'virtual', 'universe', 'galaxy'}
        self.drama_words = {'family', 'relationship', 'life', 'struggle', 'emotion', 'personal'}
        self.thriller_words = {'mystery', 'suspect', 'crime', 'detective', 'investigation', 'clue'}
        
        # Passive voice indicators
        self.passive_auxiliaries = {'is', 'are', 'was', 'were', 'been', 'be'}
        
    def extract_features(self, text):
        """Extract enhanced linguistic features for MaxEnt."""
        features = {}
        
        if pd.isna(text):
            return features
        
        text_lower = text.lower()
        
        # Tokenization
        sentences = sent_tokenize(text)
        words = word_tokenize(text_lower)
        words_clean = [w for w in words if w.isalpha()]
        
        # POS tagging
        pos_tags = pos_tag(words)
        
        # === 1. WORD FEATURES ===
        # Use word frequencies (numeric features work better with MaxEnt)
        word_freq = Counter(words_clean)
        for word, freq in word_freq.most_common(100):  # Top 100 words per document
            features[f'word_{word}'] = min(freq, 5)  # Cap at 5 to avoid extreme values
        
        # === 2. BIGRAM FEATURES (MaxEnt handles these well) ===
        bigrams = [' '.join([words_clean[i], words_clean[i+1]]) 
                   for i in range(len(words_clean)-1)]
        bigram_freq = Counter(bigrams)
        for bigram, freq in bigram_freq.most_common(20):  # Top 20 bigrams
            features[f'bigram_{bigram}'] = min(freq, 3)
        
        # === 3. MORPHOLOGICAL FEATURES ===
        # Count suffix occurrences
        for suffix in self.suffix_patterns:
            suffix_clean = suffix.replace('-', '')
            count = sum(1 for word in words_clean if word.endswith(suffix_clean))
            if count > 0:
                features[f'suffix{suffix}_count'] = min(count, 5)
        
        # Count prefix occurrences
        for prefix in self.prefix_patterns:
            prefix_clean = prefix.replace('-', '')
            count = sum(1 for word in words_clean if word.startswith(prefix_clean))
            if count > 0:
                features[f'prefix{prefix}_count'] = min(count, 5)
        
        # === 4. SYNTACTIC FEATURES ===
        # POS tag counts (numeric features)
        pos_counts = Counter([tag[:2] for _, tag in pos_tags])  # Use first 2 chars
        for pos, count in pos_counts.items():
            features[f'pos_{pos}_count'] = min(count, 10)
        
        # POS ratios (useful for MaxEnt)
        total_words = len(pos_tags)
        if total_words > 0:
            features['noun_ratio'] = pos_counts.get('NN', 0) / total_words
            features['verb_ratio'] = pos_counts.get('VB', 0) / total_words
            features['adj_ratio'] = pos_counts.get('JJ', 0) / total_words
            features['adv_ratio'] = pos_counts.get('RB', 0) / total_words
        
        # Named entity indicators
        words_original = word_tokenize(text)
        capitalized_count = sum(1 for i, w in enumerate(words_original) 
                               if i > 0 and w.isalpha() and w[0].isupper())
        if capitalized_count > 0:
            features['capitalized_count'] = min(capitalized_count, 10)
        
        # Date/year mentions
        year_matches = re.findall(r'\b(19|20)\d{2}\b', text)
        if year_matches:
            features['year_mention_count'] = min(len(year_matches), 5)
        
        # Passive voice count
        passive_count = 0
        for i in range(len(pos_tags) - 1):
            word, tag = pos_tags[i]
            next_tag = pos_tags[i + 1][1]
            if word.lower() in self.passive_auxiliaries and next_tag == 'VBN':
                passive_count += 1
        if passive_count > 0:
            features['passive_voice_count'] = min(passive_count, 5)
        
        # === 5. SEMANTIC FEATURES ===
        # Lemmatize for better matching
        lemmatized_words = set()
        for word in words_clean:
            lemma_v = self.lemmatizer.lemmatize(word, pos='v')
            lemmatized_words.add(lemma_v)
            lemma_n = self.lemmatizer.lemmatize(word, pos='n')
            lemmatized_words.add(lemma_n)
        
        # Domain-specific term counts (numeric features)
        features['action_term_count'] = len(lemmatized_words & self.action_words)
        features['romance_term_count'] = len(lemmatized_words & self.romance_words)
        features['horror_term_count'] = len(lemmatized_words & self.horror_words)
        features['comedy_term_count'] = len(lemmatized_words & self.comedy_words)
        features['scifi_term_count'] = len(lemmatized_words & self.scifi_words)
        features['drama_term_count'] = len(lemmatized_words & self.drama_words)
        features['thriller_term_count'] = len(lemmatized_words & self.thriller_words)
        
        # Sentiment counts
        features['positive_word_count'] = len(lemmatized_words & self.positive_words)
        features['negative_word_count'] = len(lemmatized_words & self.negative_words)
        
        # Sentiment ratio
        if len(words_clean) > 0:
            features['sentiment_ratio'] = (features['positive_word_count'] - 
                                          features['negative_word_count']) / len(words_clean)
        
        # === 6. STRUCTURAL FEATURES ===
        features['text_length'] = min(len(text), 1000) / 1000  # Normalized
        features['word_count'] = min(len(words_clean), 500) / 500  # Normalized
        features['sentence_count'] = min(len(sentences), 50) / 50  # Normalized
        
        if len(sentences) > 0:
            features['avg_sentence_length'] = min(len(words_clean) / len(sentences), 50) / 50
        
        if len(words_clean) > 0:
            features['avg_word_length'] = sum(len(w) for w in words_clean) / len(words_clean) / 10
        
        # === 7. FEATURE COMBINATIONS (MaxEnt strength) ===
        # Combine domain terms with sentiment
        if features.get('action_term_count', 0) > 0 and features.get('negative_word_count', 0) > 0:
            features['action_negative_combo'] = 1
        
        if features.get('romance_term_count', 0) > 0 and features.get('positive_word_count', 0) > 0:
            features['romance_positive_combo'] = 1
        
        if features.get('horror_term_count', 0) > 0 and features.get('negative_word_count', 0) > 0:
            features['horror_negative_combo'] = 1
        
        return features


class MaxEntTextClassifier:
    """
    Maximum Entropy classifier for text classification using Logistic Regression.
    Uses L2 regularization and handles numeric features effectively.
    """
    
    def __init__(self, C=1.0, max_iter=1000, top_features=5000):
        """
        Initialize MaxEnt classifier.
        
        Args:
            C: Inverse of regularization strength (smaller = stronger regularization)
            max_iter: Maximum iterations for optimization
            top_features: Maximum number of features to use
        """
        self.C = C
        self.max_iter = max_iter
        self.top_features = top_features
        self.model = None
        self.feature_extractor = FeatureExtractor()
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
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{n_samples} documents ({100*(idx+1)/n_samples:.1f}%)")
            features = self.feature_extractor.extract_features(text)
            all_features.append(features)
        print(f"  Completed: {n_samples}/{n_samples} documents (100.0%)")
        
        # Build feature vocabulary with importance scores
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
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=42,
            verbose=1,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.model.fit(X_matrix, y_encoded)
        print("Training complete!")
        
    def predict(self, X):
        """Predict class labels."""
        print(f"\nMaking predictions on {len(X)} test documents...")
        predictions = []
        
        # Extract features
        all_features = []
        for idx, text in enumerate(X):
            if (idx + 1) % 500 == 0:
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


def load_data(data_dir='../data'):
    """Load CSV files from data directory."""
    all_data = []
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        genre = csv_file.replace('.csv', '')
        filepath = os.path.join(data_dir, csv_file)
        
        try:
            df = pd.read_csv(filepath)
            if 'description' in df.columns:
                df_subset = df[['description']].copy()
                df_subset['genre'] = genre
                all_data.append(df_subset)
                print(f"Loaded {len(df_subset)} samples from {csv_file} (genre: {genre})")
            else:
                print(f"Warning: 'description' column not found in {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=['description'])
    
    print(f"\nTotal samples: {len(combined_df)}")
    print(f"Genre distribution:\n{combined_df['genre'].value_counts()}")
    
    return combined_df


def plot_confusion_matrix(y_true, y_pred, classes, save_path='../results/movies-2/confusion_matrix_Movies_maxent.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Maximum Entropy with Linguistic Features', fontsize=16, pad=20)
    plt.ylabel('True Genre', fontsize=12)
    plt.xlabel('Predicted Genre', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def evaluate_classifier(y_true, y_pred, classes, save_path='../results/movies-2/maxent_evaluation_metrics.txt'):
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
    print("="*80)
    print("MAXIMUM ENTROPY MOVIE GENRE CLASSIFICATION")
    print("WITH ENHANCED LINGUISTICALLY-MOTIVATED FEATURES")
    print("="*80)
    
    print("\n1. Loading data...")
    df = load_data('../data')
    
    print("\n2. Splitting data (70% train, 30% test)...")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df_shuffled['description'].tolist()
    y = df_shuffled['genre'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    print("\n3. Training Maximum Entropy classifier...")
    # C=1.0 provides moderate regularization; adjust if needed
    classifier = MaxEntTextClassifier(C=1.0, max_iter=1000, top_features=5000)
    classifier.fit(X_train, y_train)
    
    print("\n4. Making predictions...")
    y_pred = classifier.predict(X_test)
    
    print("\n5. Evaluating performance...")
    metrics = evaluate_classifier(y_test, y_pred, sorted(set(y_test)))
    
    print("\n6. Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, sorted(set(y_test)))
    
    print("\n7. Top features per genre:")
    top_features = classifier.get_top_features(n=10)
    for genre, features in sorted(top_features.items()):
        print(f"\n{genre}:")
        for feature, weight in features[:5]:
            print(f"  {feature}: {weight:.4f}")
    
    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETE!")
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Macro F1: {metrics['f1_macro']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
