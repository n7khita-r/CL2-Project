import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk import pos_tag, word_tokenize
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


class NewsFeatureExtractor:
    """
    Extract linguistically-motivated features for news classification.
    All features are binary (presence/absence) to maintain independence.
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        # Morphological patterns
        self.suffix_patterns = ['-tion', '-ness', '-ly', '-ment', '-ing', '-ed', '-ful', '-ous', '-ive', '-able']
        self.prefix_patterns = ['un-', 're-', 'dis-', 'non-', 'over-']
        
        # News-specific domain terminology
        self.politics_words = {'election', 'vote', 'president', 'congress', 'senate', 'bill', 
                              'policy', 'government', 'political', 'campaign', 'democrat', 'republican'}
        
        self.business_words = {'market', 'stock', 'company', 'business', 'economy', 'trade',
                              'revenue', 'profit', 'investment', 'financial', 'corporate', 'ceo'}
        
        self.tech_words = {'technology', 'software', 'app', 'digital', 'computer', 'internet',
                          'tech', 'data', 'online', 'cyber', 'AI', 'device'}
        
        self.sports_words = {'game', 'team', 'player', 'win', 'score', 'league', 'season',
                            'coach', 'championship', 'tournament', 'match', 'athlete'}
        
        self.entertainment_words = {'movie', 'film', 'show', 'music', 'star', 'celebrity',
                                   'actor', 'director', 'album', 'concert', 'performance'}
        
        self.world_news_words = {'country', 'international', 'foreign', 'nation', 'global',
                                'minister', 'embassy', 'treaty', 'diplomacy', 'border'}
        
        self.crime_words = {'police', 'arrest', 'crime', 'murder', 'shooting', 'killed',
                           'charged', 'suspect', 'investigation', 'victim', 'court'}
        
        # Sentiment/tone indicators
        self.urgent_words = {'breaking', 'urgent', 'alert', 'emergency', 'crisis', 'critical'}
        self.positive_words = {'win', 'success', 'celebrate', 'achievement', 'victory', 'amazing',
                              'great', 'excellent', 'wonderful', 'love'}
        self.negative_words = {'death', 'killed', 'disaster', 'tragedy', 'failure', 'crisis',
                              'scandal', 'controversy', 'attack', 'violence'}
        
        # Passive voice indicators
        self.passive_auxiliaries = {'is', 'are', 'was', 'were', 'been', 'be'}
        
    def extract_features(self, text):
        """Extract all features from text as binary presence/absence."""
        features = {}
        
        if pd.isna(text) or not isinstance(text, str):
            return features
        
        text_lower = text.lower()
        
        # Tokenization
        words = word_tokenize(text_lower)
        words_clean = [w for w in words if w.isalpha()]
        
        # POS tagging
        pos_tags = pos_tag(words)
        
        # === 1. WORD UNIGRAMS ===
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
        # POS tag presence (major categories)
        pos_set = set([tag for _, tag in pos_tags])
        major_pos = ['NN', 'VB', 'JJ', 'RB', 'PRP', 'IN', 'DT', 'NNP']  # Added NNP for proper nouns
        for tag_prefix in major_pos:
            if any(tag.startswith(tag_prefix) for tag in pos_set):
                features[f'has_pos_{tag_prefix}'] = 1
        
        # Named entity indicators (proper nouns are common in news)
        if any(tag.startswith('NNP') for _, tag in pos_tags):
            features['has_proper_noun'] = 1
        
        # Numbers/dates (important in news)
        if any(tag == 'CD' for _, tag in pos_tags):  # Cardinal numbers
            features['has_number'] = 1
        
        # Passive voice (common in news reporting)
        for i in range(len(pos_tags) - 1):
            word, tag = pos_tags[i]
            next_tag = pos_tags[i + 1][1]
            if word.lower() in self.passive_auxiliaries and next_tag == 'VBN':
                features['has_passive_voice'] = 1
                break
        
        # === 4. SEMANTIC FEATURES ===
        # Lemmatize for better domain matching
        lemmatized_words = set()
        for word in words_clean:
            lemma_v = self.lemmatizer.lemmatize(word, pos='v')
            lemmatized_words.add(lemma_v)
            lemma_n = self.lemmatizer.lemmatize(word, pos='n')
            lemmatized_words.add(lemma_n)
        
        # Domain-specific terminology
        if lemmatized_words & self.politics_words:
            features['has_politics_terms'] = 1
        if lemmatized_words & self.business_words:
            features['has_business_terms'] = 1
        if lemmatized_words & self.tech_words:
            features['has_tech_terms'] = 1
        if lemmatized_words & self.sports_words:
            features['has_sports_terms'] = 1
        if lemmatized_words & self.entertainment_words:
            features['has_entertainment_terms'] = 1
        if lemmatized_words & self.world_news_words:
            features['has_world_news_terms'] = 1
        if lemmatized_words & self.crime_words:
            features['has_crime_terms'] = 1
        
        # Tone/urgency indicators
        if lemmatized_words & self.urgent_words:
            features['has_urgent_tone'] = 1
        if lemmatized_words & self.positive_words:
            features['has_positive_tone'] = 1
        if lemmatized_words & self.negative_words:
            features['has_negative_tone'] = 1
        
        return features


class NaiveBayesTextClassifier:
    """
    Naive Bayes classifier for text classification.
    All features are binary with Laplace smoothing.
    """
    
    def __init__(self, alpha=1.0, top_features=3000):
        self.alpha = alpha
        self.top_features = top_features
        self.class_priors = {}
        self.feature_probs = {}
        self.feature_set = set()
        self.classes = []
        self.feature_extractor = NewsFeatureExtractor()
        
    def fit(self, X, y):
        """Train the Naive Bayes classifier."""
        print("Extracting features from training data...")
        self.classes = list(set(y))
        n_samples = len(y)
        
        # Class priors: P(class)
        class_counts = Counter(y)
        for cls in self.classes:
            self.class_priors[cls] = class_counts[cls] / n_samples
        
        # Extract features
        all_features = []
        print(f"Processing {n_samples} training documents...")
        for idx, text in enumerate(X):
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1}/{n_samples} documents ({100*(idx+1)/n_samples:.1f}%)")
            features = self.feature_extractor.extract_features(text)
            all_features.append(features)
        print(f"  Completed: {n_samples}/{n_samples} documents (100.0%)")
        
        # Build feature vocabulary (top features by document frequency)
        feature_counter = Counter()
        for features in all_features:
            feature_counter.update(features.keys())
        
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
        
        # P(feature=1|class) with Laplace smoothing
        for cls in self.classes:
            self.feature_probs[cls] = {}
            total_docs = class_doc_counts[cls]
            
            for feature in self.feature_set:
                count = class_feature_counts[cls][feature]
                # Laplace smoothing: (count + alpha) / (total + 2*alpha)
                self.feature_probs[cls][feature] = (count + self.alpha) / (total_docs + 2 * self.alpha)
    
    def predict(self, X):
        """Predict class labels."""
        print(f"Making predictions on {len(X)} test documents...")
        predictions = []
        for idx, text in enumerate(X):
            if (idx + 1) % 5000 == 0:
                print(f"  Predicted {idx + 1}/{len(X)} documents ({100*(idx+1)/len(X):.1f}%)")
            scores = self.predict_proba_single(text)
            predictions.append(max(scores, key=scores.get))
        print(f"  Completed: {len(X)}/{len(X)} documents (100.0%)")
        return predictions
    
    def predict_proba_single(self, text):
        """Predict log probabilities for single document."""
        features = self.feature_extractor.extract_features(text)
        class_scores = {}
        
        for cls in self.classes:
            # Start with log prior: log P(class)
            log_prob = np.log(self.class_priors[cls])
            
            # Add log likelihoods: log P(features|class)
            for feature in self.feature_set:
                if features.get(feature, 0) == 1:
                    # Feature present
                    log_prob += np.log(self.feature_probs[cls][feature])
                else:
                    # Feature absent
                    log_prob += np.log(1 - self.feature_probs[cls][feature])
            
            class_scores[cls] = log_prob
        
        return class_scores


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


def plot_confusion_matrix(y_true, y_pred, classes, save_path='../results/news/naivebayes_confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # For many categories, use smaller font and larger figure
    n_classes = len(classes)
    figsize = (max(12, n_classes * 0.6), max(10, n_classes * 0.5))
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, 
                cbar_kws={'label': 'Count'}, annot_kws={'size': 8})
    plt.title('Confusion Matrix - News Category Classification', fontsize=14, pad=15)
    plt.ylabel('True Category', fontsize=11)
    plt.xlabel('Predicted Category', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def evaluate_classifier(y_true, y_pred, classes, save_path='../results/news/naivebayes_evaluation_metrics.txt'):
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
    print("="*90)
    print("NAIVE BAYES NEWS CATEGORY CLASSIFICATION")
    print("WITH LINGUISTICALLY-MOTIVATED FEATURES")
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
    
    print("\n3. Training Naive Bayes classifier...")
    classifier = NaiveBayesTextClassifier(alpha=1.0, top_features=3000)
    classifier.fit(X_train, y_train)
    
    print(f"\nNumber of classes: {len(classifier.classes)}")
    
    print("\n4. Making predictions...")
    y_pred = classifier.predict(X_test)
    
    print("\n5. Evaluating performance...")
    metrics = evaluate_classifier(y_test, y_pred, sorted(classifier.classes))
    
    print("\n6. Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, sorted(classifier.classes))
    
    print("\n" + "="*90)
    print("CLASSIFICATION COMPLETE!")
    print("="*90)


if __name__ == "__main__":
    main()