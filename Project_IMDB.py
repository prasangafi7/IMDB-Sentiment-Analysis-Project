import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy import stats

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
    
    print("OVERVIEW TEXT PREPROCESSING")
    print("=" * 60)
    
    # CLEAN AND PREPROCESS OVERVIEW
    # ========================================================================
    print("\n1. TEXT PREPROCESSING")
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
    print("\n\n2. OVERVIEW LENGTH AND CORRELATION ANALYSIS")
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
    
    # Plot 1: Length vs IMDB Rating (using filtered data)
    ax1.scatter(df_valid['Length_Words'], df_valid['IMDB_Rating'], alpha=0.5)
    z = np.polyfit(df_valid['Length_Words'], df_valid['IMDB_Rating'], 1)
    p = np.poly1d(z)
    ax1.plot(df_valid['Length_Words'], p(df_valid['Length_Words']), "r--", alpha=0.8)
    ax1.set_xlabel('Overview Length (words)')
    ax1.set_ylabel('IMDB Rating')
    ax1.set_title(f'Length vs IMDB Rating (r={corr_rating:.3f}, p={p_rating:.3f})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Length vs Meta Score
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
    
    print("\nPreprocessing Complete!")
    return df


if __name__ == '__main__':
    csv_path = 'imdb_top_1000.csv'
    df = exploratory_data_analysis(csv_path, plot=True)
    genre_analysis(csv_path)
    df_processed = overview_text_preprocessing(df)

    