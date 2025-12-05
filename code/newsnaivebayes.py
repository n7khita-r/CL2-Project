"""
Naive Bayes category Classification with Independent Feature Groups
Optimized for Terminal/Local Execution
"""

# ============================================================================
# SECTION 1: INSTALL DEPENDENCIES
# ============================================================================
print("Checking dependencies...")
import sys
import subprocess

def install_package(package):
    """Install package if not already installed."""
    try:
        __import__(package.split('[')[0])
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Install required packages
packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'nltk', 'spacy']
for pkg in packages:
    install_package(pkg)

# Download spaCy model
try:
    import spacy
    spacy.load('en_core_web_sm')
except:
    print("Downloading spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Download NLTK data
import nltk
print("Downloading NLTK data...")
for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 
                 'averaged_perceptron_tagger_eng', 'wordnet', 'stopwords', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass

print("All dependencies ready!\n")

# ============================================================================
# SECTION 2: IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import os
import re
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for terminal
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Load spaCy model for NER
print("Loading spaCy NER model...")
nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
print("spaCy model loaded!\n")

# ============================================================================
# SECTION 3: FEATURE EXTRACTOR
# ============================================================================
class IndependentFeatureExtractor:
    """
    Extract features designed for Naive Bayes conditional independence assumption.
    Features are organized into mutually exclusive groups.
    """
    
    def __init__(self):
        # Function words (closed class)
        self.function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'of', 'to', 'in', 'on', 'at', 'by',
            'for', 'with', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
            'may', 'might', 'must', 'can', 'shall', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'not', 'no', 'yes', 'so', 'than', 'too', 'very'
        }
        
        self.stopwords = set(stopwords.words('english'))
        
        # POS tag categories
        self.pos_categories = {
            'NOUN': ['NN', 'NNS', 'NNP', 'NNPS'],
            'VERB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'ADJ': ['JJ', 'JJR', 'JJS'],
            'ADV': ['RB', 'RBR', 'RBS'],
            'PRON': ['PRP', 'PRP$', 'WP', 'WP$'],
            'DET': ['DT', 'PDT', 'WDT'],
            'PREP': ['IN'],
            'CONJ': ['CC'],
            'NUM': ['CD']
        }
        
        # Common prefixes and suffixes
        self.prefixes = ['un', 'in', 'dis', 'mis', 're', 'pre', 'post', 'anti', 'de', 
                        'over', 'under', 'out', 'sub', 'inter', 'fore', 'super', 
                        'semi', 'non', 'mid', 'ex']
        self.suffixes = ['ing', 'ed', 'ly', 'ness', 'ment', 'tion', 'sion', 'ful', 
                        'less', 'able', 'ible', 'ous', 'ious', 'ive', 'er', 'est', 
                        'ish', 'al', 'ity', 'ty']
        
    def extract_features(self, text):
        """
        Extract features organized into independent groups.
        """
        features = {
            'content_words': {},      # GROUP 1: Content word presence (binary)
            'pos_tags': {},           # GROUP 2: POS distributions
            'structure': {},          # GROUP 3: Document structure
            'punctuation': {},        # GROUP 4: Punctuation patterns
            'ner': {},                # GROUP 5: Named entities
            'lexical': {},            # GROUP 6: Lexical statistics
            'affixes': {}             # GROUP 7: Prefix/suffix presence
        }
        
        if pd.isna(text) or not text.strip():
            return features
        
        text_lower = text.lower()
        
        # Tokenization
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text_lower)
        words = [t for t in tokens if t.isalpha()]
        
        if len(words) == 0:
            return features
        
        total_words = len(words)
        
        # POS tagging
        pos_tags = pos_tag(words)
        
        # ====================================================================
        # GROUP 1: CONTENT WORD PRESENCE (BINARY)
        # ====================================================================
        content_words = [w for w in words if w not in self.function_words 
                        and w not in self.stopwords and len(w) > 2]
        
        content_word_freq = Counter(content_words)
        for word, count in content_word_freq.most_common(100):
            features['content_words'][f'word_has_{word}'] = 1
        
        # ====================================================================
        # GROUP 2: POS TAG COUNTS
        # ====================================================================
        pos_counts = Counter([tag for _, tag in pos_tags])
        
        for category, tags in self.pos_categories.items():
            count = sum(pos_counts.get(tag, 0) for tag in tags)
            features['pos_tags'][f'pos_count_{category.lower()}'] = count
        
        # ====================================================================
        # GROUP 3: DOCUMENT STRUCTURE
        # ====================================================================
        length_bin = min(total_words // 100, 10)
        features['structure'][f'length_bin_{length_bin}'] = 1
        
        sent_count = len(sentences)
        sent_bin = min(sent_count // 5, 10)
        features['structure'][f'sent_bin_{sent_bin}'] = 1
        
        if sent_count > 0:
            avg_sent_len = total_words / sent_count
            avg_bin = min(int(avg_sent_len // 5), 10)
            features['structure'][f'avg_sent_len_bin_{avg_bin}'] = 1
        
        # ====================================================================
        # GROUP 4: PUNCTUATION COUNTS
        # ====================================================================
        features['punctuation']['punct_period'] = text.count('.')
        features['punctuation']['punct_question'] = text.count('?')
        features['punctuation']['punct_exclaim'] = text.count('!')
        features['punctuation']['punct_comma'] = text.count(',')
        features['punctuation']['punct_semicolon'] = text.count(';')
        features['punctuation']['punct_colon'] = text.count(':')
        features['punctuation']['punct_dash'] = text.count('--') + text.count('—')
        features['punctuation']['punct_quotes'] = text.count('"') + text.count("'")
        features['punctuation']['punct_ellipsis'] = text.count('...')
        
        # ====================================================================
        # GROUP 5: NAMED ENTITY COUNTS
        # ====================================================================
        doc = nlp(text[:100000])
        entity_counts = Counter([ent.label_ for ent in doc.ents])
        
        for ent_type in ['PERSON', 'ORG', 'GPE', 'DATE', 'TIME', 'MONEY', 
                        'PERCENT', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LOC',
                        'FAC', 'NORP', 'LAW', 'LANGUAGE', 'CARDINAL', 'ORDINAL']:
            count = entity_counts.get(ent_type, 0)
            features['ner'][f'ner_{ent_type.lower()}'] = count
        
        # ====================================================================
        # GROUP 6: LEXICAL STATISTICS
        # ====================================================================
        unique_words = len(set(words))
        ttr = unique_words / total_words if total_words > 0 else 0
        ttr_bin = min(int(ttr * 10), 9)
        features['lexical'][f'ttr_bin_{ttr_bin}'] = 1
        
        fw_count = sum(1 for w in words if w in self.function_words)
        fw_ratio = fw_count / total_words if total_words > 0 else 0
        fw_bin = min(int(fw_ratio * 10), 9)
        features['lexical'][f'fw_ratio_bin_{fw_bin}'] = 1
        
        # ====================================================================
        # GROUP 7: PREFIX/SUFFIX PRESENCE (BINARY)
        # ====================================================================
        for prefix in self.prefixes:
            has_prefix = any(word.startswith(prefix) and len(word) > len(prefix) + 2 
                           for word in content_words)
            if has_prefix:
                features['affixes'][f'prefix_{prefix}'] = 1
        
        for suffix in self.suffixes:
            has_suffix = any(word.endswith(suffix) and len(word) > len(suffix) + 2 
                           for word in content_words)
            if has_suffix:
                features['affixes'][f'suffix_{suffix}'] = 1
        
        return features


# ============================================================================
# SECTION 4: NAIVE BAYES CLASSIFIER
# ============================================================================
class NaiveBayesTextClassifier:
    """
    Naive Bayes classifier with feature group ablation support.
    """
    
    def __init__(self, alpha=1.0, feature_groups=None):
        """
        Args:
            alpha: Smoothing parameter for Naive Bayes
            feature_groups: List of feature groups to use. If None, uses all.
        """
        self.alpha = alpha
        self.model = None
        self.feature_extractor = IndependentFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
        if feature_groups is None:
            self.feature_groups = ['content_words', 'pos_tags', 'structure', 
                                  'punctuation', 'ner', 'lexical', 'affixes']
        else:
            self.feature_groups = feature_groups
        
        print(f"\n{'='*60}")
        print(f"ACTIVE FEATURE GROUPS:")
        for group in self.feature_groups:
            print(f"  ✓ {group}")
        print(f"{'='*60}\n")
        
    def _features_to_vector(self, features_dict):
        """Convert feature dictionary to numpy array."""
        vector = []
        for fname in self.feature_names:
            found = False
            for group in self.feature_groups:
                if fname in features_dict.get(group, {}):
                    vector.append(features_dict[group][fname])
                    found = True
                    break
            if not found:
                vector.append(0)
        return np.array(vector)
    
    def fit(self, X, y):
        """Train the classifier."""
        print("Extracting features from training data...")
        n_samples = len(y)
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        all_features = []
        print(f"Processing {n_samples} training documents...")
        for idx, text in enumerate(X):
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{n_samples} ({100*(idx+1)/n_samples:.1f}%)")
            features = self.feature_extractor.extract_features(text)
            all_features.append(features)
        print(f"  Completed: {n_samples}/{n_samples} (100.0%)")
        
        print(f"\nBuilding feature vocabulary from active groups...")
        feature_counter = Counter()
        for features in all_features:
            for group in self.feature_groups:
                if group in features:
                    feature_counter.update(features[group].keys())
        
        max_features = 5000
        if len(feature_counter) > max_features:
            self.feature_names = [f for f, _ in feature_counter.most_common(max_features)]
        else:
            self.feature_names = list(feature_counter.keys())
        
        print(f"Total unique features: {len(feature_counter)}")
        print(f"Using {len(self.feature_names)} features")
        
        for group in self.feature_groups:
            group_features = [f for f in self.feature_names if 
                            any(f in feat_dict.get(group, {}) 
                                for feat_dict in all_features)]
            print(f"  - {group}: {len(group_features)} features")
        
        print("\nConverting to feature matrix...")
        X_matrix = np.zeros((n_samples, len(self.feature_names)))
        
        for i, features in enumerate(all_features):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{n_samples} ({100*(i+1)/n_samples:.1f}%)")
            X_matrix[i] = self._features_to_vector(features)
        
        print(f"\nFeature matrix shape: {X_matrix.shape}")
        
        print("\nTraining Multinomial Naive Bayes model...")
        self.model = MultinomialNB(alpha=self.alpha)
        self.model.fit(X_matrix, y_encoded)
        
        train_pred = self.model.predict(X_matrix)
        train_acc = accuracy_score(y_encoded, train_pred)
        print(f"\nTraining accuracy: {train_acc:.4f}")
        print("Training complete!")
        
    def predict(self, X):
        """Predict class labels."""
        print(f"\nMaking predictions on {len(X)} test documents...")
        
        all_features = []
        for idx, text in enumerate(X):
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(X)} ({100*(idx+1)/len(X):.1f}%)")
            features = self.feature_extractor.extract_features(text)
            all_features.append(features)
        print(f"  Completed: {len(X)}/{len(X)} (100.0%)")
        
        print("\nConverting to feature matrix...")
        X_matrix = np.zeros((len(X), len(self.feature_names)))
        
        for i, features in enumerate(all_features):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(X)} ({100*(i+1)/len(X):.1f}%)")
            X_matrix[i] = self._features_to_vector(features)
        
        y_pred_encoded = self.model.predict(X_matrix)
        predictions = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return predictions
    
    def get_top_features(self, n=10):
        """Get top features for each class based on log probabilities."""
        if self.model is None:
            return None
        
        top_features = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            log_probs = self.model.feature_log_prob_[i]
            top_indices = np.argsort(log_probs)[-n:][::-1]
            top_features[class_name] = [
                (self.feature_names[idx], log_probs[idx]) 
                for idx in top_indices
            ]
        
        return top_features


# ============================================================================
# SECTION 5: DATA LOADING
# ============================================================================
def load_data(data_dir='../data', json_filename='News_Category_Dataset_v3.json', 
              samples_per_category=3000, min_samples_threshold=500):
    """
    Load news data from JSON file with balanced sampling per category.
    
    Args:
        data_dir: Directory containing the JSON file
        json_filename: Name of the JSON file
        samples_per_category: Target number of samples per category
        min_samples_threshold: Minimum samples a category must have to be included
    
    Returns:
        pandas.DataFrame with 'description' and 'category' columns
    """
    all_data = []
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        print("Please ensure your data directory exists and contains the JSON file.")
        sys.exit(1)
    
    filepath = os.path.join(data_dir, json_filename)
    
    # Check if JSON file exists
    if not os.path.exists(filepath):
        print(f"Error: JSON file '{json_filename}' not found in '{data_dir}'")
        print(f"Looking for: {filepath}")
        sys.exit(1)
    
    print(f"Loading data from {json_filename}...")
    print(f"Target: {samples_per_category} samples per category\n")
    
    try:
        # Read JSON file line by line (JSONL format)
        data_records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    data_records.append(record)
                except json.JSONDecodeError:
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(data_records)
        
        print(f"✓ Loaded {len(df)} total records")
        print(f"\nAvailable columns: {list(df.columns)}")
        
        # Check for required columns
        if 'category' not in df.columns:
            print("Error: 'category' column not found in the dataset")
            sys.exit(1)
        
        # Use short_description if available, otherwise use headline
        if 'short_description' in df.columns:
            text_column = 'short_description'
        elif 'headline' in df.columns:
            text_column = 'headline'
        else:
            print("Error: Neither 'short_description' nor 'headline' column found")
            sys.exit(1)
        
        print(f"Using '{text_column}' as text content\n")
        
        # Create standardized dataframe
        df_clean = df[[text_column, 'category']].copy()
        df_clean.columns = ['description', 'category']
        
        # Remove rows with missing descriptions
        df_clean = df_clean.dropna(subset=['description'])
        df_clean = df_clean[df_clean['description'].str.strip() != '']
        
        print(f"After removing empty descriptions: {len(df_clean)} records\n")
        
        # Get category distribution
        category_counts = df_clean['category'].value_counts()
        print("Original category distribution:")
        print(category_counts)
        print()
        
        # Filter categories with minimum threshold
        valid_categories = category_counts[category_counts >= min_samples_threshold].index
        df_filtered = df_clean[df_clean['category'].isin(valid_categories)]
        
        print(f"Categories with at least {min_samples_threshold} samples: {len(valid_categories)}")
        print()
        
        # Balance sampling per category
        balanced_data = []
        for category in sorted(valid_categories):
            category_df = df_filtered[df_filtered['category'] == category].copy()
            
            if len(category_df) > samples_per_category:
                category_df = category_df.sample(n=samples_per_category, random_state=42)
                print(f"✓ Sampled {samples_per_category} from '{category}' (had {len(df_filtered[df_filtered['category'] == category])})")
            else:
                print(f"✓ Using all {len(category_df)} samples from '{category}'")
            
            balanced_data.append(category_df)
        
        # Combine all balanced data
        combined_df = pd.concat(balanced_data, ignore_index=True)
        
        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print(f"BALANCED DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {len(combined_df)}")
        print(f"Number of categories: {combined_df['category'].nunique()}")
        print(f"\nCategory distribution:")
        print(combined_df['category'].value_counts().sort_index())
        print(f"{'='*60}\n")
        
        return combined_df
        
    except Exception as e:
        print(f"✗ Error loading {json_filename}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# SECTION 6: EVALUATION & VISUALIZATION
# ============================================================================
def plot_confusion_matrix(y_true, y_pred, classes, save_path='../results/NaiveBayes/News/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Naive Bayes with Independent Features', 
              fontsize=14, pad=15)
    plt.ylabel('True category', fontsize=11)
    plt.xlabel('Predicted category', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {save_path}")
    plt.close()


def evaluate_classifier(y_true, y_pred, classes, save_path='../results/NaiveBayes/News/evaluation_metrics.txt'):
    """Calculate and save evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    output_lines = []
    output_lines.append("="*80)
    output_lines.append("NAIVE BAYES CLASSIFIER - EVALUATION METRICS")
    output_lines.append("="*80)
    output_lines.append(f"\nOverall Accuracy: {accuracy:.4f}\n")
    output_lines.append("-"*80)
    output_lines.append(f"{'category':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    output_lines.append("-"*80)
    
    for i, cls in enumerate(classes):
        output_lines.append(f"{cls:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10.0f}")
    
    output_lines.append("-"*80)
    output_lines.append(f"{'Macro Average':<20} {p_macro:<12.4f} {r_macro:<12.4f} {f1_macro:<12.4f}")
    output_lines.append("="*80)
    
    print("\n" + "\n".join(output_lines))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("\n".join(output_lines))
    print(f"\n✓ Evaluation metrics saved to: {save_path}")
    
    return {'accuracy': accuracy, 'f1_macro': f1_macro}


# ============================================================================
# SECTION 7: MAIN PIPELINE
# ============================================================================
def main():
    print("\n" + "="*80)
    print("NAIVE BAYES category CLASSIFICATION")
    print("WITH INDEPENDENT FEATURE GROUPS")
    print("="*80)
    
    print("\n1. Loading balanced data from ../data...")
    df = load_data('../data', samples_per_category=1000)
    
    print("\n2. Splitting data (70% train, 30% test) with stratification...")
    X = df['description'].tolist()
    y = df['category'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    active_groups = [
        'content_words',
        'pos_tags',
        'structure',
        'punctuation',
        'ner',
        'lexical',
        'affixes'
    ]
    
    print("\n3. Training classifier...")
    classifier = NaiveBayesTextClassifier(alpha=1.0, feature_groups=active_groups)
    classifier.fit(X_train, y_train)
    
    print("\n4. Making predictions...")
    y_pred = classifier.predict(X_test)
    
    print("\n5. Evaluating performance...")
    classes = sorted(set(y_test))
    metrics = evaluate_classifier(y_test, y_pred, classes)
    
    print("\n6. Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, classes)
    
    print("\n7. Top features per category (by log probability):")
    top_features = classifier.get_top_features(n=10)
    
    features_output = []
    features_output.append("="*80)
    features_output.append("TOP FEATURES PER category (by log probability)")
    features_output.append("="*80)
    
    for category in sorted(top_features.keys()):
        print(f"\n{category.upper()}:")
        features_output.append(f"\n{category.upper()}:")
        for feature, log_prob in top_features[category][:5]:
            line = f"  {feature:30s} {log_prob:+.4f}"
            print(line)
            features_output.append(line)
    
    features_path = '../results/NaiveBayes/News/top_features.txt'
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    with open(features_path, 'w') as f:
        f.write("\n".join(features_output))
    print(f"\n✓ Top features saved to: {features_path}")
    
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'correct': [t == p for t, p in zip(y_test, y_pred)]
    })
    predictions_path = '../results/NaiveBayes/News/predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions saved to: {predictions_path}")
    
    summary_path = '../results/NaiveBayes/News/summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model Type: Multinomial Naive Bayes\n")
        f.write(f"Smoothing (alpha): {classifier.alpha}\n")
        f.write(f"Active Feature Groups: {', '.join(classifier.feature_groups)}\n")
        f.write(f"Total Features: {len(classifier.feature_names)}\n")
        f.write(f"Number of Classes: {len(classes)}\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Testing Samples: {len(X_test)}\n\n")
        f.write(f"Final Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Final Macro F1: {metrics['f1_macro']:.4f}\n")
        f.write("="*80 + "\n")
    print(f"✓ Model summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print(f"FINAL ACCURACY: {metrics['accuracy']:.4f}")
    print(f"FINAL MACRO F1: {metrics['f1_macro']:.4f}")
    print("="*80)
    print(f"\n✓ All results saved to ../results/NaiveBayes/News/")


# ============================================================================
# RUN THE PIPELINE
# ============================================================================
if __name__ == "__main__":
    main()
