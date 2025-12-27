import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

REQUIRED_COLUMNS = [
    'Series_Title', 'Genre', 'IMDB_Rating', 'Runtime', 'Gross', 'Released_Year', 'Overview', 'Meta_score'
]


def load_and_preprocess(csv_file: str) -> pd.DataFrame:
    """Load CSV and perform safe preprocessing used by analysis functions.

    - Ensures numeric conversions use `errors='coerce'`.
    - Adds `Runtime_Minutes`, `Gross_Earnings`, and numeric `Year` columns.
    - Leaves original columns intact.
    """
    df = pd.read_csv(csv_file)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Safe numeric conversions
    df['Runtime_Minutes'] = pd.to_numeric(
        df['Runtime'].str.replace(' min', '', regex=False), errors='coerce'
    )
    df['Gross_Earnings'] = pd.to_numeric(df['Gross'].str.replace(',', '', regex=False), errors='coerce')
    df['Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')

    # Tokenize genres into lists (empty list for missing)
    df['Genre_List'] = df['Genre'].apply(lambda x: x.split(', ') if pd.notna(x) else [])

    return df


def exploratory_data_analysis(csv_file: str, plot: bool = True, save_fig: bool = True, fig_path: str = 'eda_results.png') -> pd.DataFrame:
    """Perform exploratory data analysis and optionally plot trends over time.

    Returns the preprocessed DataFrame so other functions can reuse it.
    """
    df = load_and_preprocess(csv_file)

    print("\nEXPLORATORY DATA ANALYSIS - IMDB TOP 1000 MOVIES")
    print("=" * 60)
    print(f"Total number of movies: {len(df)}")

    # Basic distributions
    print("\nIMDB Rating Distribution:")
    print(df['IMDB_Rating'].describe())

    print("\nRuntime Distribution (in minutes):")
    print(df['Runtime_Minutes'].describe())

    print("\nGross Earnings Distribution:")
    print(df['Gross_Earnings'].describe())
    print(f"Note: {df['Gross_Earnings'].isna().sum()} movies have missing gross earnings data")

    if plot:
        print("\nCREATING VISUALIZATIONS FOR TRENDS OVER TIME")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('IMDB Movies - Trends Over Time', fontsize=16, fontweight='bold')

        yearly_rating = df.groupby('Year', dropna=True)['IMDB_Rating'].mean()
        axes[0].plot(yearly_rating.index, yearly_rating.values, marker='o', color='green', linewidth=2)
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Average IMDB Rating')
        axes[0].set_title('Average IMDB Rating by Year')
        axes[0].grid(True, alpha=0.3)

        yearly_gross = df.groupby('Year', dropna=True)['Gross_Earnings'].mean() / 1_000_000
        axes[1].plot(yearly_gross.index, yearly_gross.values, marker='o', color='orange', linewidth=2)
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Average Gross (Millions $)')
        axes[1].set_title('Box Office Gross by Year')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_fig:
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved as '{fig_path}'")
        plt.show()

    print("\nSUMMARY STATISTICS")
    print(f"Average IMDB Rating: {df['IMDB_Rating'].mean():.2f}")
    print(f"Average Runtime: {df['Runtime_Minutes'].mean():.1f} minutes")
    print(f"Average Gross Earnings: ${df['Gross_Earnings'].mean()/1_000_000:.2f} Million")

    if df['Year'].dropna().empty:
        print("Year Range: N/A")
    else:
        print(f"Year Range: {int(df['Year'].min())} to {int(df['Year'].max())}")

    return df


def genre_analysis(csv_file: str, top_n: int = 10, save_fig: bool = True) -> None:
    """Perform genre analysis and visualizations.

    - Uses `explode` + `groupby` for efficient aggregation.
    - Prints top genres, top combinations, and average ratings.
    """
    df = load_and_preprocess(csv_file)

    print("\nGENRE ANALYSIS")
    print("=" * 60)

    # Explode genres for efficient aggregation
    exploded = df.explode('Genre_List')
    exploded['Genre_List'] = exploded['Genre_List'].replace('', np.nan)

    # Top individual genres
    genre_counts = exploded['Genre_List'].value_counts(dropna=True)
    print(f"\nTop {top_n} Most Common Genres:")
    for genre, count in genre_counts.head(top_n).items():
        print(f"  {genre}: {count} movies")

    # Top exact combinations (original Genre string)
    print(f"\nTop {top_n} Genre Combinations:")
    combos = df['Genre'].value_counts().head(top_n)
    for combo, count in combos.items():
        print(f"  {combo}: {count} movies")

    # Average ratings by genre (using exploded)
    print(f"\nAverage IMDB Rating by Genre (Top {top_n}):")
    ratings = exploded.groupby('Genre_List', dropna=True)['IMDB_Rating'].mean()
    ratings = ratings.dropna()
    ratings = ratings.sort_values(ascending=False)
    for genre, rating in ratings.head(top_n).items():
        print(f"  {genre}: {rating:.2f}")

    # Popularity over decades (pivot table)
    print("\nGENRE POPULARITY OVER DECADES")
    df['Decade'] = (df['Year'] // 10) * 10
    exploded_decade = df.explode('Genre_List')
    exploded_decade['Genre_List'] = exploded_decade['Genre_List'].replace('', np.nan)
    pivot = exploded_decade.pivot_table(index='Decade', columns='Genre_List', values='Series_Title', aggfunc='count', fill_value=0)

    # Select top 5 genres by overall count
    top5 = genre_counts.head(5).index.tolist()
    decades = sorted(pivot.index.dropna().tolist())

    plt.figure(figsize=(12, 6))
    for genre in top5:
        if genre in pivot.columns:
            plt.plot(decades, pivot.loc[decades, genre], marker='o', linewidth=2, label=genre)

    plt.xlabel('Decade', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.title('Genre Popularity Over Decades', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_fig:
        plt.savefig('genre_popularity_decades.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'genre_popularity_decades.png'")
    plt.show()


# Overview Text Preprocessing
# ===========================================

def overview_text_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses Overview text and analyzes correlation with ratings.
    Accepts preprocessed DataFrame from load_and_preprocess().
    """
    df = df.copy()
    
    print("\n\nOVERVIEW TEXT PREPROCESSING")
    print("=" * 60)
    
    # CLEAN AND PREPROCESS OVERVIEW
    # ========================================================================
    print("\nTEXT PREPROCESSING")
    print("-" * 60)
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        # Lowercase
        text = text.lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        cleaned = [lemmatizer.lemmatize(word) for word in tokens 
                   if word.isalpha() and word not in stop_words]
        return ' '.join(cleaned)
    
    df['Overview_Cleaned'] = df['Overview'].apply(preprocess_text) 
    
    # COMPUTE LENGTH AND CORRELATION
    # ========================================================================
    print("\n\nOVERVIEW LENGTH AND CORRELATION ANALYSIS")
    print("-" * 60)
    
    # Calculate lengths
    df['Length_Words'] = df['Overview_Cleaned'].apply(lambda x: len(x.split()))
    df['Length_Chars'] = df['Overview_Cleaned'].apply(lambda x: len(x))
    
    print("\nAverage Overview Length:")
    print(f"  Words: {df['Length_Words'].mean():.2f}")
    print(f"  Characters: {df['Length_Chars'].mean():.2f}")
    
    # Filter out zero-length overviews for correlation
    df_valid = df[df['Length_Words'] > 0].copy()
    print(f"\nValid overviews for correlation: {len(df_valid)}/{len(df)}")
    
    # Correlation analysis with p-values
    corr_rating, p_rating = stats.pearsonr(df_valid['Length_Words'], df_valid['IMDB_Rating'])
    
    df_meta_valid = df_valid.dropna(subset=['Meta_score'])
    if len(df_meta_valid) > 0:
        corr_meta, p_meta = stats.pearsonr(df_meta_valid['Length_Words'], df_meta_valid['Meta_score'])
    else:
        corr_meta, p_meta = np.nan, np.nan
    
    print("\nCorrelation with Ratings (Pearson):")
    print(f"  Overview Length vs IMDB Rating: {corr_rating:.4f} (p={p_rating:.4f})")
    print(f"  Overview Length vs Meta Score: {corr_meta:.4f} (p={p_meta:.4f})")
    print(f"  Meta Score data available: {len(df_meta_valid)}/{len(df)} rows")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Length vs IMDB Rating (using filtered data)
    ax1.scatter(df_valid['Length_Words'], df_valid['IMDB_Rating'], alpha=0.5)
    z = np.polyfit(df_valid['Length_Words'], df_valid['IMDB_Rating'], 1)
    p = np.poly1d(z)
    ax1.plot(df_valid['Length_Words'], p(df_valid['Length_Words']), "r--", alpha=0.8)
    ax1.set_xlabel('Overview Length (words)')
    ax1.set_ylabel('IMDB Rating')
    ax1.set_title(f'Length vs IMDB Rating (r={corr_rating:.3f}, p={p_rating:.3f})')
    ax1.grid(True, alpha=0.3)
    
    # Plot Length vs Meta Score
    if len(df_meta_valid) > 0:
        ax2.scatter(df_meta_valid['Length_Words'], df_meta_valid['Meta_score'], alpha=0.5, color='orange')
        z2 = np.polyfit(df_meta_valid['Length_Words'], df_meta_valid['Meta_score'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(df_meta_valid['Length_Words'], p2(df_meta_valid['Length_Words']), "r--", alpha=0.8)
        ax2.set_title(f'Length vs Meta Score (r={corr_meta:.3f}, p={p_meta:.3f})')
    else:
        ax2.text(0.5, 0.5, 'Insufficient Meta Score data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Length vs Meta Score (No data)')
    ax2.set_xlabel('Overview Length (words)')
    ax2.set_ylabel('Meta Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overview_correlation.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'overview_correlation.png'")
    plt.show()
    
    return df


def keyword_extraction_tfidf(df: pd.DataFrame) -> None:
    """
    Extracts top keywords using TF-IDF and compares high vs low rated movies.
    Accepts preprocessed DataFrame from load_and_preprocess().
    """
    df = df.dropna(subset=['Overview']).copy()
    
    print("\n\nTF-IDF KEYWORD EXTRACTION")
    print("=" * 60)
    
    # EXTRACT TOP KEYWORDS FROM MOVIE OVERVIEWS
    # ========================================================================
    print("\n1. TOP KEYWORDS FROM ALL MOVIES")
    print("-" * 60)
    
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Overview'])
    
    feature_names = tfidf.get_feature_names_out()
    avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    
    keywords = pd.DataFrame({
        'keyword': feature_names,
        'score': avg_scores
    }).sort_values('score', ascending=False)
    
    print("\nTop 15 Keywords:")
    for idx, row in keywords.head(15).iterrows():
        print(f"  {row['keyword']}: {row['score']:.4f}")
    
    # COMPARE HIGH-RATED VS LOW-RATED MOVIES
    # ========================================================================
    print("\n\n2. COMPARING HIGH-RATED VS LOW-RATED MOVIES")
    print("-" * 60)
    
    high_rated = df[df['IMDB_Rating'] > 8.0]
    low_rated = df[df['IMDB_Rating'] < 8.0]
    
    print(f"\nHigh-rated (>8.0): {len(high_rated)} movies")
    print(f"Low-rated (<8.0): {len(low_rated)} movies")
    
    # High-rated keywords
    print("\nTop 10 Keywords - High-Rated Movies:")
    try:
        tfidf_high = TfidfVectorizer(
            max_features=50,
            stop_words='english',
            min_df=1,
            max_df=0.8,
            ngram_range=(1, 1)
        )
        matrix_high = tfidf_high.fit_transform(high_rated['Overview'])
        
        high_keywords = pd.DataFrame({
            'keyword': tfidf_high.get_feature_names_out(),
            'score': np.mean(matrix_high.toarray(), axis=0)
        }).sort_values('score', ascending=False)
        
        for idx, row in high_keywords.head(10).iterrows():
            print(f"  {row['keyword']}: {row['score']:.4f}")
    except ValueError as e:
        print(f"  Error: {e}")
        print("  Unable to extract keywords from high-rated movies.")
        high_keywords = None
    
    # Low-rated keywords
    print("\nTop 10 Keywords - Low-Rated Movies:")
    try:
        tfidf_low = TfidfVectorizer(
            max_features=50,
            stop_words='english',
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 1)
        )
        matrix_low = tfidf_low.fit_transform(low_rated['Overview'])
        
        low_keywords = pd.DataFrame({
            'keyword': tfidf_low.get_feature_names_out(),
            'score': np.mean(matrix_low.toarray(), axis=0)
        }).sort_values('score', ascending=False)
        
        for idx, row in low_keywords.head(10).iterrows():
            print(f"  {row['keyword']}: {row['score']:.4f}")
    except ValueError as e:
        print(f"  Error: {e}")
        print("  The dataset has too few low-rated movies (<8.0) with meaningful text.")
        low_keywords = None
    
    # Visualization
    if high_keywords is not None and low_keywords is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        top_high = high_keywords.head(10)
        ax1.barh(top_high['keyword'], top_high['score'], color='green', alpha=0.7)
        ax1.set_xlabel('TF-IDF Score')
        ax1.set_title('High-Rated Movies (>8.0)')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        top_low = low_keywords.head(10)
        ax2.barh(top_low['keyword'], top_low['score'], color='red', alpha=0.7)
        ax2.set_xlabel('TF-IDF Score')
        ax2.set_title('Low-Rated Movies (<8.0)')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('tfidf_keywords.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'tfidf_keywords.png'")
        plt.show()
    else:
        print("\nSkipping visualization due to insufficient low-rated movie data.")
    
    print("\nAnalysis Complete!")

# Run the function

if __name__ == '__main__':
    csv_path = 'imdb_top_1000.csv'
    df = exploratory_data_analysis(csv_path, plot=True)
    genre_analysis(csv_path)
    df_processed = overview_text_preprocessing(df)
    keyword_extraction_tfidf(df_processed)

    