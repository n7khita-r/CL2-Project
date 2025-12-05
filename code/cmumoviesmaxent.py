"""
Maximum Entropy Genre Classification with Linguistically-Motivated Features
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
packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'nltk', 'spacy', 'scipy']
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
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, vstack
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for terminal
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from nltk import pos_tag, word_tokenize, sent_tokenize, ngrams
from nltk.corpus import stopwords

# Load spaCy model for NER
print("Loading spaCy NER model...")
nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
print("spaCy model loaded!\n")

# ============================================================================
# SECTION 3: FEATURE EXTRACTOR
# ============================================================================
class LinguisticFeatureExtractor:
    """
    Extract linguistically-motivated features for MaxEnt classification.
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
        """Extract linguistic features."""
        features = {}
        
        if pd.isna(text) or not text.strip():
            return features
        
        text_lower = text.lower()
        
        # Tokenization
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text_lower)
        words = [t for t in tokens if t.isalpha()]
        
        if len(words) == 0:
            return features
        
        # POS tagging
        pos_tags = pos_tag(words)
        
        # ====================================================================
        # FEATURE GROUP 1: FUNCTION WORD FREQUENCIES
        # ====================================================================
        function_word_counts = Counter()
        for word in words:
            if word in self.function_words:
                function_word_counts[word] += 1
        
        total_words = len(words)
        for fw in ['the', 'a', 'of', 'to', 'and', 'in', 'is', 'was', 'it', 'that',
                   'for', 'on', 'with', 'as', 'be', 'at', 'by', 'i', 'this', 'but']:
            features[f'fw_{fw}'] = function_word_counts.get(fw, 0) / total_words
        
        features['function_word_ratio'] = sum(function_word_counts.values()) / total_words
        
        # ====================================================================
        # FEATURE GROUP 2: POS TAG DISTRIBUTIONS
        # ====================================================================
        pos_counts = Counter([tag for _, tag in pos_tags])
        
        for category, tags in self.pos_categories.items():
            count = sum(pos_counts.get(tag, 0) for tag in tags)
            features[f'pos_prop_{category.lower()}'] = count / total_words
        
        # ====================================================================
        # FEATURE GROUP 3: CONTENT WORD UNIGRAMS
        # ====================================================================
        content_words = [w for w in words if w not in self.function_words 
                        and w not in self.stopwords and len(w) > 2]
        
        content_word_freq = Counter(content_words)
        
        for word, _ in content_word_freq.most_common(100):
            features[f'word_has_{word}'] = 1
        
        # ====================================================================
        # FEATURE GROUP 4: DOCUMENT LENGTH & SENTENCE LENGTH STATISTICS
        # ====================================================================
        features['doc_length_tokens'] = min(total_words / 500, 1.0)
        features['sentence_count'] = min(len(sentences) / 50, 1.0)
        
        if len(sentences) > 0:
            sent_lengths = [len(word_tokenize(s)) for s in sentences]
            features['mean_sent_length'] = np.mean(sent_lengths) / 50
            features['median_sent_length'] = np.median(sent_lengths) / 50
            
            if len(sent_lengths) > 1:
                features['stddev_sent_length'] = np.std(sent_lengths) / 20
            
            features['prop_short_sent'] = sum(1 for l in sent_lengths if l < 10) / len(sentences)
            features['prop_long_sent'] = sum(1 for l in sent_lengths if l > 30) / len(sentences)
        
        # ====================================================================
        # FEATURE GROUP 5: TYPE-TOKEN RATIO & LEXICAL RICHNESS
        # ====================================================================
        unique_words = len(set(words))
        features['type_token_ratio'] = unique_words / total_words if total_words > 0 else 0
        
        # ====================================================================
        # FEATURE GROUP 6: PUNCTUATION & ORTHOGRAPHIC FEATURES
        # ====================================================================
        all_tokens = word_tokenize(text)
        
        features['punct_period'] = text.count('.') / total_words
        features['punct_question'] = text.count('?') / total_words
        features['punct_exclaim'] = text.count('!') / total_words
        features['punct_comma'] = text.count(',') / total_words
        features['punct_semicolon'] = text.count(';') / total_words
        features['punct_colon'] = text.count(':') / total_words
        features['punct_dash'] = (text.count('--') + text.count('—')) / total_words
        features['punct_quotes'] = (text.count('"') + text.count("'")) / total_words
        features['punct_ellipsis'] = text.count('...') / total_words
        
        # Capitalization features
        words_original = [t for t in word_tokenize(text) if t.isalpha()]
        if words_original:
            allcaps_count = sum(1 for w in words_original if w.isupper() and len(w) > 1)
            features['allcaps_ratio'] = allcaps_count / len(words_original)
            
            titlecase_count = sum(1 for i, w in enumerate(words_original) 
                                 if i > 0 and w[0].isupper() and not w.isupper())
            features['titlecase_ratio'] = titlecase_count / len(words_original)
        
        # ====================================================================
        # FEATURE GROUP 7: NAMED ENTITY RECOGNITION TAGS
        # ====================================================================
        doc = nlp(text[:100000])
        
        entity_counts = Counter([ent.label_ for ent in doc.ents])
        total_entities = sum(entity_counts.values())
        
        if total_entities > 0:
            for ent_type in ['PERSON', 'ORG', 'GPE', 'DATE', 'TIME', 'MONEY', 
                            'PERCENT', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LOC',
                            'FAC', 'NORP', 'LAW', 'LANGUAGE', 'CARDINAL', 'ORDINAL']:
                count = entity_counts.get(ent_type, 0)
                features[f'ner_{ent_type.lower()}'] = count / total_entities
            
            features['ner_density'] = total_entities / total_words
        else:
            features['ner_density'] = 0
        
        # ====================================================================
        # FEATURE GROUP 8: PREFIX/SUFFIX FEATURES (MORPHOLOGICAL)
        # ====================================================================
        for prefix in self.prefixes:
            count = sum(1 for word in content_words 
                       if word.startswith(prefix) and len(word) > len(prefix) + 2)
            features[f'prefix_{prefix}_count'] = count / total_words if total_words > 0 else 0
        
        for suffix in self.suffixes:
            count = sum(1 for word in content_words 
                       if word.endswith(suffix) and len(word) > len(suffix) + 2)
            features[f'suffix_{suffix}_count'] = count / total_words if total_words > 0 else 0

        # ====================================================================
        # FEATURE GROUP 9: WORD N-GRAMS (BIGRAMS) - CORRELATED FEATURE
        # ====================================================================
        # Capture local word order and common phrases. 
        if len(content_words) >= 2:
            bigrams = list(ngrams(content_words, 2))
            bigram_freq = Counter(bigrams)
            
            # Add top 50 most common bigrams as features.
            # This helps identify collocations specific to certain categories.
            for bigram, _ in bigram_freq.most_common(50):
                features[f'bigram_has_{"_".join(bigram)}'] = 1
        
        return features


# ============================================================================
# SECTION 4: MAXIMUM ENTROPY CLASSIFIER
# ============================================================================
class MaxEntTextClassifier:
    """Maximum Entropy classifier using Logistic Regression with sparse matrices."""
    
    def __init__(self, C=1.0, max_iter=500):
        self.C = C
        self.max_iter = max_iter
        self.model = None
        self.feature_extractor = LinguisticFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def _features_to_vector(self, features_dict):
        """Convert feature dictionary to numpy array."""
        return np.array([features_dict.get(f, 0) for f in self.feature_names])
        
    def fit(self, X, y):
        """Train the classifier using memory-efficient processing."""
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
        
        print("\nBuilding feature vocabulary...")
        feature_counter = Counter()
        for features in all_features:
            feature_counter.update(features.keys())
        
        max_features = 5000
        if len(feature_counter) > max_features:
            non_word_features = [f for f in feature_counter.keys() if not f.startswith('word_has_')]
            word_features = [f for f in feature_counter.keys() if f.startswith('word_has_')]
            
            word_freq = {f: feature_counter[f] for f in word_features}
            top_word_features = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:3000]
            
            self.feature_names = non_word_features + top_word_features
            self.feature_names = self.feature_names[:max_features]
        else:
            self.feature_names = list(feature_counter.keys())
        
        print(f"Total unique features: {len(feature_counter)}")
        print(f"Using {len(self.feature_names)} features (memory optimized)")
        
        print("\nConverting to feature matrix (using sparse format)...")
        batch_size = 5000
        sparse_matrices = []
        
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_features = all_features[batch_start:batch_end]
            
            batch_matrix = np.zeros((len(batch_features), len(self.feature_names)))
            for i, features in enumerate(batch_features):
                batch_matrix[i] = self._features_to_vector(features)
            
            sparse_matrices.append(csr_matrix(batch_matrix))
            print(f"  Processed batch {batch_start//batch_size + 1}/{(n_samples-1)//batch_size + 1}")
            
            del batch_matrix
        
        X_matrix = vstack(sparse_matrices)
        del sparse_matrices
        
        print(f"Sparse feature matrix shape: {X_matrix.shape}")
        print(f"Memory usage: ~{X_matrix.data.nbytes / (1024**2):.1f} MB")
        print(f"Sparsity: {100 * (1 - X_matrix.nnz / (X_matrix.shape[0] * X_matrix.shape[1])):.1f}% zeros")
        
        print(f"\nFeature matrix stats:")
        print(f"  Non-zero elements: {X_matrix.nnz}")
        print(f"  Min value: {X_matrix.data.min():.4f}")
        print(f"  Max value: {X_matrix.data.max():.4f}")
        print(f"  Mean value: {X_matrix.data.mean():.4f}")
        
        print("\nTraining Maximum Entropy model...")
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=42,
            verbose=1,
            n_jobs=-1
        )
        
        self.model.fit(X_matrix, y_encoded)
        
        train_pred = self.model.predict(X_matrix)
        train_acc = accuracy_score(y_encoded, train_pred)
        print(f"\nTraining accuracy: {train_acc:.4f}")
        print("Training complete!")
        
    def predict(self, X):
        """Predict class labels using memory-efficient processing."""
        print(f"\nMaking predictions on {len(X)} test documents...")
        
        all_features = []
        for idx, text in enumerate(X):
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(X)} ({100*(idx+1)/len(X):.1f}%)")
            features = self.feature_extractor.extract_features(text)
            all_features.append(features)
        print(f"  Completed: {len(X)}/{len(X)} (100.0%)")
        
        print("\nConverting to feature matrix...")
        batch_size = 5000
        sparse_matrices = []
        
        for batch_start in range(0, len(X), batch_size):
            batch_end = min(batch_start + batch_size, len(X))
            batch_features = all_features[batch_start:batch_end]
            
            batch_matrix = np.zeros((len(batch_features), len(self.feature_names)))
            for i, features in enumerate(batch_features):
                batch_matrix[i] = self._features_to_vector(features)
            
            sparse_matrices.append(csr_matrix(batch_matrix))
            del batch_matrix
        
        X_matrix = vstack(sparse_matrices)
        del sparse_matrices
        
        y_pred_encoded = self.model.predict(X_matrix)
        predictions = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return predictions
    
    def get_top_features(self, n=10):
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


# ============================================================================
# SECTION 5: DATA LOADING
# ============================================================================
def load_data(data_dir='../data', csv_filename='movies_processed.csv', 
              samples_per_genre=3000, min_samples_threshold=100):
    """
    Load movie data from CSV file with balanced sampling per genre.
    Uses the FIRST genre from the genres column as the gold standard label.
    
    Args:
        data_dir: Directory containing the CSV file
        csv_filename: Name of the CSV file
        samples_per_genre: Target number of samples per genre
        min_samples_threshold: Minimum samples a genre must have to be included
    
    Returns:
        pandas.DataFrame with 'description' and 'genre' columns
    """
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        print("Please ensure your data directory exists and contains the CSV file.")
        sys.exit(1)
    
    filepath = os.path.join(data_dir, csv_filename)
    
    # Check if CSV file exists
    if not os.path.exists(filepath):
        print(f"Error: CSV file '{csv_filename}' not found in '{data_dir}'")
        print(f"Looking for: {filepath}")
        sys.exit(1)
    
    print(f"Loading data from {csv_filename}...")
    print(f"Target: {samples_per_genre} samples per genre\n")
    
    try:
        # Read CSV file
        df = pd.read_csv(filepath)
        
        print(f"✓ Loaded {len(df)} total records")
        print(f"\nAvailable columns: {list(df.columns)}")
        
        # Check for required columns
        if 'summary' not in df.columns:
            print("Error: 'summary' column not found in the dataset")
            sys.exit(1)
        
        if 'genres' not in df.columns:
            print("Error: 'genres' column not found in the dataset")
            sys.exit(1)
        
        print(f"\nProcessing movie summaries and extracting first genre...\n")
        
        # Create standardized dataframe with first genre only
        df_clean = df[['summary', 'genres']].copy()
        
        # Remove rows with missing summaries or genres
        df_clean = df_clean.dropna(subset=['summary', 'genres'])
        df_clean = df_clean[df_clean['summary'].str.strip() != '']
        df_clean = df_clean[df_clean['genres'].str.strip() != '']
        
        # Extract first genre from pipe-separated list
        def extract_first_genre(genres_str):
            """Extract the first genre from a pipe-separated string."""
            if pd.isna(genres_str) or genres_str.strip() == '':
                return None
            genres_list = str(genres_str).split('|')
            return genres_list[0].strip() if genres_list else None
        
        df_clean['genre'] = df_clean['genres'].apply(extract_first_genre)
        
        # Remove rows where genre extraction failed
        df_clean = df_clean.dropna(subset=['genre'])
        df_clean = df_clean[df_clean['genre'] != '']
        
        # Rename summary to description for consistency
        df_clean = df_clean[['summary', 'genre']].copy()
        df_clean.columns = ['description', 'genre']
        
        print(f"After cleaning: {len(df_clean)} records with valid summaries and genres\n")
        
        # Get genre distribution
        genre_counts = df_clean['genre'].value_counts()
        print("Original genre distribution (using first genre only):")
        print(genre_counts)
        print(f"\nTotal unique genres: {len(genre_counts)}")
        print()
        
        # Filter genres with minimum threshold
        valid_genres = genre_counts[genre_counts >= min_samples_threshold].index
        df_filtered = df_clean[df_clean['genre'].isin(valid_genres)]
        
        print(f"Genres with at least {min_samples_threshold} samples: {len(valid_genres)}")
        print(f"Genres included: {sorted(valid_genres.tolist())}")
        print()
        
        # Balance sampling per genre
        balanced_data = []
        for genre in sorted(valid_genres):
            genre_df = df_filtered[df_filtered['genre'] == genre].copy()
            
            if len(genre_df) > samples_per_genre:
                genre_df = genre_df.sample(n=samples_per_genre, random_state=42)
                print(f"✓ Sampled {samples_per_genre} from '{genre}' (had {len(df_filtered[df_filtered['genre'] == genre])})")
            else:
                print(f"✓ Using all {len(genre_df)} samples from '{genre}'")
            
            balanced_data.append(genre_df)
        
        # Combine all balanced data
        combined_df = pd.concat(balanced_data, ignore_index=True)
        
        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print(f"BALANCED DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {len(combined_df)}")
        print(f"Number of genres: {combined_df['genre'].nunique()}")
        print(f"\nGenre distribution:")
        print(combined_df['genre'].value_counts().sort_index())
        print(f"{'='*60}\n")
        
        return combined_df
        
    except Exception as e:
        print(f"✗ Error loading {csv_filename}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# SECTION 6: EVALUATION & VISUALIZATION
# ============================================================================
def plot_confusion_matrix(y_true, y_pred, classes, save_path='../results/MaxEnt/CMUMovies/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - MaxEnt with Linguistic Features', 
              fontsize=14, pad=15)
    plt.ylabel('True Genre', fontsize=11)
    plt.xlabel('Predicted Genre', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {save_path}")
    plt.close()


def evaluate_classifier(y_true, y_pred, classes, save_path='../results/MaxEnt/CMUMovies/evaluation_metrics.txt'):
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
    output_lines.append("MAXIMUM ENTROPY CLASSIFIER - EVALUATION METRICS")
    output_lines.append("="*80)
    output_lines.append(f"\nOverall Accuracy: {accuracy:.4f}\n")
    output_lines.append("-"*80)
    output_lines.append(f"{'Genre':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
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
    print("MAXIMUM ENTROPY GENRE CLASSIFICATION")
    print("WITH BALANCED SAMPLING AND AFFIX FEATURES")
    print("="*80)
    
    print("\n1. Loading balanced data from ../data...")
    df = load_data('../data', samples_per_genre=3000)
    
    print("\n2. Splitting data (70% train, 30% test) with stratification...")
    X = df['description'].tolist()
    y = df['genre'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    train_dist = pd.Series(y_train).value_counts().sort_index()
    test_dist = pd.Series(y_test).value_counts().sort_index()
    
    print(f"\nTrain distribution:")
    for genre, count in train_dist.items():
        print(f"  {genre:15s}: {count:4d} samples")
    
    print(f"\nTest distribution:")
    for genre, count in test_dist.items():
        print(f"  {genre:15s}: {count:4d} samples")
    print(f"{'='*60}\n")
    
    print("\n3. Training classifier...")
    classifier = MaxEntTextClassifier(C=1.0, max_iter=500)
    classifier.fit(X_train, y_train)
    
    print("\n4. Making predictions...")
    y_pred = classifier.predict(X_test)
    
    print("\n5. Evaluating performance...")
    classes = sorted(set(y_test))
    metrics = evaluate_classifier(y_test, y_pred, classes)
    
    print("\n6. Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, classes)
    
    print("\n7. Top features per genre (by absolute weight):")
    top_features = classifier.get_top_features(n=10)
    
    features_output = []
    features_output.append("="*80)
    features_output.append("TOP FEATURES PER GENRE (by absolute weight)")
    features_output.append("="*80)
    
    for genre in sorted(top_features.keys()):
        print(f"\n{genre.upper()}:")
        features_output.append(f"\n{genre.upper()}:")
        for feature, weight in top_features[genre][:5]:
            line = f"  {feature:30s} {weight:+.4f}"
            print(line)
            features_output.append(line)
    
    features_path = '../results/MaxEnt/CMUMovies/top_features.txt'
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    with open(features_path, 'w') as f:
        f.write("\n".join(features_output))
    print(f"\n✓ Top features saved to: {features_path}")
    
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'correct': [t == p for t, p in zip(y_test, y_pred)]
    })
    predictions_path = '../results/MaxEnt/CMUMovies/predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions saved to: {predictions_path}")
    
    summary_path = '../results/MaxEnt/CMUMovies/summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model Type: Maximum Entropy (Logistic Regression)\n")
        f.write(f"Regularization (C): {classifier.C}\n")
        f.write(f"Max Iterations: {classifier.max_iter}\n")
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
    print(f"\n✓ All results saved to results/CMUmaxent/")


# ============================================================================
# RUN THE PIPELINE
# ============================================================================
if __name__ == "__main__":
	main()
