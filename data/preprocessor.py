import json
import csv

def preprocess_movie_data(plot_file='plot_summaries.txt', 
                          metadata_file='movie.metadata.tsv',
                          output_file='movies_processed.csv'):
    """
    Combine plot summaries with movie metadata for genre prediction task.
    
    Args:
        plot_file: Path to plot_summaries.txt
        metadata_file: Path to movie.metadata.tsv
        output_file: Path to output CSV file
    """
    
    # Step 1: Read plot summaries into a dictionary
    print("Reading plot summaries...")
    plots = {}
    with open(plot_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                movie_id, summary = parts
                plots[movie_id] = summary
    
    print(f"Loaded {len(plots)} plot summaries")
    
    # Step 2: Read metadata and combine with plots
    print("Reading metadata and combining...")
    processed_movies = []
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 9:
                movie_id = parts[0]
                
                # Only process movies that have plot summaries
                if movie_id in plots:
                    # Extract genre dictionary
                    genre_dict = parts[8] if len(parts) > 8 else '{}'
                    
                    # Parse genres from JSON-like format
                    genres = []
                    try:
                        genre_obj = json.loads(genre_dict)
                        genres = list(genre_obj.values())
                    except:
                        pass
                    
                    # Skip movies without genres
                    if not genres:
                        continue
                    
                    processed_movies.append({
                        'movie_id': movie_id,
                        'title': parts[2] if len(parts) > 2 else '',
                        'release_year': parts[3] if len(parts) > 3 else '',
                        'genres': '|'.join(genres),  # Multiple genres separated by |
                        'num_genres': len(genres),
                        'summary': plots[movie_id]
                    })
    
    print(f"Processed {len(processed_movies)} movies with both summaries and genres")
    
    # Step 3: Write to CSV
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        if processed_movies:
            writer = csv.DictWriter(f, fieldnames=processed_movies[0].keys())
            writer.writeheader()
            writer.writerows(processed_movies)
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total movies processed: {len(processed_movies)}")
    
    all_genres = []
    for movie in processed_movies:
        all_genres.extend(movie['genres'].split('|'))
    
    unique_genres = set(all_genres)
    print(f"Unique genres: {len(unique_genres)}")
    
    # Count genre frequency
    from collections import Counter
    genre_counts = Counter(all_genres)
    print(f"\nTop 10 most common genres:")
    for genre, count in genre_counts.most_common(10):
        print(f"  {genre}: {count}")
    
    print(f"\nOutput saved to: {output_file}")
    return processed_movies


if __name__ == "__main__":
    # Run the preprocessing
    movies = preprocess_movie_data()
    
    # Show a sample
    if movies:
        print("\n=== Sample Record ===")
        sample = movies[0]
        for key, value in sample.items():
            if key == 'summary':
                print(f"{key}: {value[:200]}...")  # Truncate summary
            else:
                print(f"{key}: {value}")
