import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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
    All features are binary (presence/absence) to maintain independence.
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        # Morphological patterns
        self.suffix_patterns = ['-tion', '-ness', '-ly', '-ment', '-ing', '-ed', '-er', '-est', 
                                '-ful', '-less', '-ous', '-ive', '-able', '-ible', '-al', '-ial']
        self.prefix_patterns = ['un-', 're-', 'pre-', 'anti-', 'dis-', 'mis-', 'non-', 'over-', 
                                'under-', 'super-', 'sub-', 'inter-', 'trans-']
        
        # Sentiment indicators (simple lexicon)
        self.positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                               'love', 'best', 'beautiful', 'perfect', 'happy', 'joy', 'brilliant'}
        self.negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor',
                               'sad', 'angry', 'evil', 'dark', 'death', 'kill', 'murder', 'fear'}
        
        # Domain-specific terminology for movie genres
        self.action_words = {'fight', 'battle', 'war', 'mission', 'escape', 'chase', 'explosion',
                            'weapon', 'soldier', 'agent', 'rescue', 'attack', 'destroy'}
        self.romance_words = {'love', 'relationship', 'romance', 'marry', 'wedding', 'heart',
                             'kiss', 'couple', 'affair', 'passionate', 'desire'}
        self.horror_words = {'horror', 'ghost', 'haunted', 'dead', 'death', 'blood', 'murder',
                            'monster', 'terror', 'nightmare', 'evil', 'demon', 'zombie'}
        self.comedy_words = {'comedy', 'funny', 'laugh', 'humor', 'hilarious', 'joke', 'amusing'}
        self.scifi_words = {'space', 'future', 'alien', 'robot', 'technology', 'planet', 'galaxy',
                           'time', 'science', 'experiment', 'dimension', 'virtual'}
        
        # Passive voice indicators
        self.passive_auxiliaries = {'is', 'are', 'was', 'were', 'been', 'be', 'being'}
        
    def extract_features(self, text):
        """
        Extract all features from text as binary presence/absence.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of binary features
        """
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
        
        # === BASELINE FEATURES ===
        
        # 1. Unigrams and bigrams (top TF-IDF features only - added later)
        for word in words_clean:
            features[f'word_{word}'] = 1
        
        # 2. Character n-grams (2-5) - presence for each unique n-gram
        for n in range(2, 6):
            for word in words_clean:
                if len(word) >= n:
                    for i in range(len(word) - n + 1):
                        char_ngram = word[i:i+n]
                        features[f'char_{n}gram_{char_ngram}'] = 1
        
        # 3. Document statistics - REMOVED (not linguistically motivated)
        
        # === MORPHOLOGICAL FEATURES ===
        
        # 1. Suffix patterns (presence/absence)
        for suffix in self.suffix_patterns:
            suffix_clean = suffix.replace('-', '')
            has_suffix = any(word.endswith(suffix_clean) for word in words_clean)
            features[f'has_suffix{suffix}'] = 1 if has_suffix else 0
        
        # 2. Prefix patterns (presence/absence)
        for prefix in self.prefix_patterns:
            prefix_clean = prefix.replace('-', '')
            has_prefix = any(word.startswith(prefix_clean) for word in words_clean)
            features[f'has_prefix{prefix}'] = 1 if has_prefix else 0
        
        # 3. Lemma-based features (unique lemmas)
        lemmas = set()
        for word in words_clean:
            try:
                lemma = self.lemmatizer.lemmatize(word, pos='v')
                lemmas.add(lemma)
                features[f'lemma_{lemma}'] = 1
            except:
                pass
        
        # === SYNTACTIC FEATURES ===
        
        # 1. POS tag distribution (presence of each tag type)
        pos_counts = Counter([tag for _, tag in pos_tags])
        for tag in pos_counts:
            features[f'has_pos_{tag}'] = 1
        
        # POS bigrams
        for i in range(len(pos_tags) - 1):
            bigram = f"{pos_tags[i][1]}_{pos_tags[i+1][1]}"
            features[f'pos_bigram_{bigram}'] = 1
        
        # 2. Named entity indicators (simple rule-based)
        # PERSON: capitalized words not at sentence start
        # ORG: sequences of capitalized words
        # DATE: year patterns, month names
        
        words_original = word_tokenize(text)
        for i, word in enumerate(words_original):
            if word[0].isupper() and i > 0:
                features['has_named_entity'] = 1
                break
        
        # Year pattern (19xx, 20xx)
        if re.search(r'\b(19|20)\d{2}\b', text):
            features['has_date_year'] = 1
        
        # Month names
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december']
        if any(month in text_lower for month in months):
            features['has_date_month'] = 1
        
        # 3. Passive voice indicators
        for i in range(len(pos_tags) - 1):
            word, tag = pos_tags[i]
            next_word, next_tag = pos_tags[i + 1]
            
            # Pattern: auxiliary + past participle (VBN)
            if word.lower() in self.passive_auxiliaries and next_tag == 'VBN':
                features['has_passive_voice'] = 1
                break
        
        # 4. Sentence structure complexity (binned)
        if sentences:
            # Average words per sentence
            avg_words_per_sent = len(words_clean) / len(sentences)
            features['complex_sentences'] = 1 if avg_words_per_sent > 20 else 0
            features['simple_sentences'] = 1 if avg_words_per_sent <= 20 else 0
        
        # === SEMANTIC FEATURES ===
        
        # 1. Domain-specific terminology (presence/absence)
        features['has_action_terms'] = 1 if any(w in words_clean for w in self.action_words) else 0
        features['has_romance_terms'] = 1 if any(w in words_clean for w in self.romance_words) else 0
        features['has_horror_terms'] = 1 if any(w in words_clean for w in self.horror_words) else 0
        features['has_comedy_terms'] = 1 if any(w in words_clean for w in self.comedy_words) else 0
        features['has_scifi_terms'] = 1 if any(w in words_clean for w in self.scifi_words) else 0
        
        # 2. Sentiment polarity indicators (presence/absence)
        features['has_positive_sentiment'] = 1 if any(w in words_clean for w in self.positive_words) else 0
        features['has_negative_sentiment'] = 1 if any(w in words_clean for w in self.negative_words) else 0
        
        # 3. Topic-indicative verb and noun patterns
        verbs = [word.lower() for word, tag in pos_tags if tag.startswith('VB')]
        nouns = [word.lower() for word, tag in pos_tags if tag.startswith('NN')]
        
        for verb in set(verbs):
            if verb.isalpha():
                features[f'verb_{verb}'] = 1
        
        for noun in set(nouns):
            if noun.isalpha():
                features[f'noun_{noun}'] = 1
        
        return features


class NaiveBayesTextClassifier:
    """
    Naive Bayes classifier for text classification with rich linguistic features.
    All features are binary (presence/absence) with Laplace smoothing.
    """
    
    def __init__(self, alpha=1.0, top_features=1000):
        """
        Initialize the Naive Bayes classifier.
        
        Args:
            alpha: Laplace smoothing parameter
            top_features: Number of top features to keep (dimensionality reduction)
        """
        self.alpha = alpha
        self.top_features = top_features
        self.class_priors = {}
        self.feature_probs = {}
        self.feature_set = set()
        self.classes = []
        self.feature_extractor = FeatureExtractor()
        
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        
        Args:
            X: List of text documents
            y: List of class labels
        """
        print("Extracting features from training data...")
        self.classes = list(set(y))
        n_samples = len(y)
        
        # Calculate class priors
        class_counts = Counter(y)
        for cls in self.classes:
            self.class_priors[cls] = class_counts[cls] / n_samples
        
        # Extract features from all documents with progress reporting
        all_features = []
        print(f"Processing {n_samples} training documents...")
        for idx, text in enumerate(X):
            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  Processed {idx + 1}/{n_samples} documents ({100*(idx+1)/n_samples:.1f}%)")
            features = self.feature_extractor.extract_features(text)
            all_features.append(features)
        print(f"  Completed: {n_samples}/{n_samples} documents (100.0%)")
        
        # Build feature vocabulary
        feature_counter = Counter()
        for features in all_features:
            feature_counter.update(features.keys())
        
        # Keep only top features (by document frequency)
        if len(feature_counter) > self.top_features:
            self.feature_set = set([f for f, _ in feature_counter.most_common(self.top_features)])
        else:
            self.feature_set = set(feature_counter.keys())
        
        print(f"Total unique features: {len(feature_counter)}")
        print(f"Using top {len(self.feature_set)} features")
        
        # Calculate feature probabilities per class
        class_feature_counts = {cls: defaultdict(int) for cls in self.classes}
        class_doc_counts = {cls: 0 for cls in self.classes}
        
        for features, label in zip(all_features, y):
            class_doc_counts[label] += 1
            for feature in self.feature_set:
                if features.get(feature, 0) == 1:
                    class_feature_counts[label][feature] += 1
        
        # Calculate P(feature|class) with Laplace smoothing
        for cls in self.classes:
            self.feature_probs[cls] = {}
            total_docs = class_doc_counts[cls]
            
            for feature in self.feature_set:
                count = class_feature_counts[cls][feature]
                # P(feature=1|class) = (count + alpha) / (total + 2*alpha)
                self.feature_probs[cls][feature] = (count + self.alpha) / (total_docs + 2 * self.alpha)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples.
        
        Args:
            X: List of text documents
            
        Returns:
            List of dictionaries mapping classes to log probabilities
        """
        predictions = []
        
        for text in X:
            predictions.append(self.predict_proba_single(text))
        
        return predictions
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Args:
            X: List of text documents
            
        Returns:
            List of predicted class labels
        """
        print(f"Making predictions on {len(X)} test documents...")
        proba = []
        for idx, text in enumerate(X):
            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  Predicted {idx + 1}/{len(X)} documents ({100*(idx+1)/len(X):.1f}%)")
            scores = self.predict_proba_single(text)
            proba.append(scores)
        print(f"  Completed: {len(X)}/{len(X)} documents (100.0%)")
        
        predictions = [max(scores, key=scores.get) for scores in proba]
        return predictions
    
    def predict_proba_single(self, text):
        """
        Predict class probabilities for a single document.
        
        Args:
            text: Text document
            
        Returns:
            Dictionary mapping classes to log probabilities
        """
        features = self.feature_extractor.extract_features(text)
        class_scores = {}
        
        for cls in self.classes:
            # Start with log prior
            log_prob = np.log(self.class_priors[cls])
            
            # Add log probabilities for features
            for feature in self.feature_set:
                if features.get(feature, 0) == 1:
                    # Feature is present
                    log_prob += np.log(self.feature_probs[cls][feature])
                else:
                    # Feature is absent
                    log_prob += np.log(1 - self.feature_probs[cls][feature])
            
            class_scores[cls] = log_prob
        
        return class_scores


def load_data(data_dir='../data'):
    """
    Load all CSV files from the data directory.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        DataFrame with descriptions and genres
    """
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


def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Naive Bayes with Linguistic Features', fontsize=16, pad=20)
    plt.ylabel('True Genre', fontsize=12)
    plt.xlabel('Predicted Genre', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def evaluate_classifier(y_true, y_pred, classes):
    """
    Calculate and display evaluation metrics.
    """
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
    
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    print("\n" + "-"*80)
    print("PER-CLASS METRICS")
    print("-"*80)
    print(f"{'Genre':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    
    for i, cls in enumerate(classes):
        print(f"{cls:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10.0f}")
    
    print("-"*80)
    print(f"{'Macro Average':<20} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}")
    print(f"{'Micro Average':<20} {precision_micro:<12.4f} {recall_micro:<12.4f} {f1_micro:<12.4f}")
    print("="*80)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro
    }


def main():
    """
    Main function to run the complete pipeline.
    """
    print("="*80)
    print("NAIVE BAYES MOVIE GENRE CLASSIFICATION")
    print("WITH LINGUISTICALLY-MOTIVATED FEATURES")
    print("="*80)
    
    print("\n1. Loading data...")
    df = load_data('../data')
    
    print("\n2. Splitting data (70% train, 30% test)...")
    # Shuffle the dataframe first to ensure random distribution
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df_shuffled['description'].tolist()
    y = df_shuffled['genre'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Verify split distribution
    train_dist = Counter(y_train)
    test_dist = Counter(y_test)
    print("\nTrain set distribution:")
    for genre, count in sorted(train_dist.items()):
        print(f"  {genre}: {count}")
    print("\nTest set distribution:")
    for genre, count in sorted(test_dist.items()):
        print(f"  {genre}: {count}")
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    print("\n3. Training Naive Bayes classifier with linguistic features...")
    classifier = NaiveBayesTextClassifier(alpha=1.0, top_features=5000)
    classifier.fit(X_train, y_train)
    
    print(f"Number of classes: {len(classifier.classes)}")
    
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
