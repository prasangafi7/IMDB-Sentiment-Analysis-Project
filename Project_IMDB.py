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
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


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

# SENTIMENT ANALYSIS USING VADER
# ===========================================

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)

def sentiment_analysis_vader(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measures sentiment polarity using VADER and explores correlation with ratings and gross.
    Accepts preprocessed DataFrame from load_and_preprocess().
    """
    df = df.dropna(subset=['Overview']).copy()
    
    print("\n\nSENTIMENT ANALYSIS - VADER")
    print("=" * 60)
    
    # MEASURE SENTIMENT POLARITY
    # ========================================================================
    print("\nMEASURING SENTIMENT POLARITY")
    print("-" * 60)
    
    # Initialize VADER
    sia = SentimentIntensityAnalyzer()
    
    # Calculate sentiment for each overview
    print("Analyzing sentiment...")
    vader_scores = []
    for overview in df['Overview']:
        scores = sia.polarity_scores(overview)
        vader_scores.append(scores['compound'])
    
    df['Sentiment'] = vader_scores
    
    print(f"\nSentiment Score Statistics:")
    print(f"  Mean: {df['Sentiment'].mean():.4f}")
    print(f"  Median: {df['Sentiment'].median():.4f}")
    print(f"  Min: {df['Sentiment'].min():.4f}")
    print(f"  Max: {df['Sentiment'].max():.4f}")
    
    # Classify sentiments
    df['Sentiment_Label'] = df['Sentiment'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )
    
    print(f"\nSentiment Distribution:")
    print(df['Sentiment_Label'].value_counts())
    
    # CORRELATION WITH IMDB RATING AND GROSS
    # ========================================================================
    print("\n\nCORRELATION ANALYSIS")
    print("-" * 60)
    
    # Clean gross data
    df['Gross_Clean'] = df['Gross'].str.replace(',', '').astype(float)
    
    # Calculate correlations with p-values
    corr_imdb, p_imdb = stats.pearsonr(df['Sentiment'], df['IMDB_Rating'])
    
    df_gross_valid = df.dropna(subset=['Gross_Clean'])
    if len(df_gross_valid) > 1:
        corr_gross, p_gross = stats.pearsonr(df_gross_valid['Sentiment'], df_gross_valid['Gross_Clean'])
    else:
        corr_gross, p_gross = np.nan, np.nan
    
    print(f"\nCorrelation Results:")
    print(f"  Sentiment vs IMDB Rating: {corr_imdb:.4f} (p={p_imdb:.4f})")
    print(f"  Sentiment vs Box Office Gross: {corr_gross:.4f} (p={p_gross:.4f})")
    print(f"  Gross data available: {len(df_gross_valid)}/{len(df)} rows")
    
    
    # VISUALIZATIONS
    # ========================================================================
    print("\n\nCREATING VISUALIZATIONS")
    print("-" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Sentiment Analysis with VADER', fontsize=16, fontweight='bold')
    
    # Plot Sentiment Distribution
    axes[0].hist(df['Sentiment'], bins=30, color='skyblue', edgecolor='black')
    axes[0].axvline(df['Sentiment'].mean(), color='red', 
                    linestyle='--', label=f"Mean: {df['Sentiment'].mean():.2f}")
    axes[0].set_xlabel('Sentiment Score')
    axes[0].set_ylabel('Number of Movies')
    axes[0].set_title('Sentiment Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Sentiment vs IMDB Rating
    axes[1].scatter(df['Sentiment'], df['IMDB_Rating'], alpha=0.5, color='green')
    z = np.polyfit(df['Sentiment'], df['IMDB_Rating'], 1)
    p = np.poly1d(z)
    axes[1].plot(df['Sentiment'], p(df['Sentiment']), "r--", alpha=0.8)
    axes[1].set_xlabel('Sentiment Score')
    axes[1].set_ylabel('IMDB Rating')
    axes[1].set_title(f'Sentiment vs IMDB Rating (r = {corr_imdb:.3f})')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Sentiment vs Gross
    df_gross = df.dropna(subset=['Gross_Clean'])
    axes[2].scatter(df_gross['Sentiment'], df_gross['Gross_Clean']/1e6, alpha=0.5, color='purple')
    axes[2].set_xlabel('Sentiment Score')
    axes[2].set_ylabel('Box Office Gross (Millions $)')
    axes[2].set_title(f'Sentiment vs Gross (r = {corr_gross:.3f})')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sentiment_correlation.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'sentiment_correlation.png'")
    plt.show()
    
    print("\nAnalysis Complete!")
    return df


def sentiment_analysis_bert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measures sentiment polarity using BERT and explores correlation with ratings and gross.
    Accepts preprocessed DataFrame from load_and_preprocess().
    """
    df = df.dropna(subset=['Overview']).copy()
    
    print("\n\nSENTIMENT ANALYSIS - BERT")
    print("=" * 60)
    
    # MEASURE SENTIMENT POLARITY WITH BERT
    # ========================================================================
    print("\nMEASURING SENTIMENT POLARITY")
    print("-" * 60)
    print("Loading BERT model (this may take a moment)...")
    
    # Initialize BERT sentiment pipeline
    bert_sentiment = pipeline('sentiment-analysis', 
                             model='distilbert-base-uncased-finetuned-sst-2-english')
    
    print(f"Analyzing {len(df)} movie overviews with BERT...")
    print("(This will take 5-10 minutes)\n")
    
    # Calculate sentiment for each overview
    bert_scores = []
    bert_labels = []
    
    for i, overview in enumerate(df['Overview']):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(df)} movies processed...")
        
        try:
            # Truncate to 512 characters (BERT limit)
            truncated = overview[:512]
            result = bert_sentiment(truncated)[0]
            
            # Convert to -1 to +1 scale
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            bert_scores.append(score)
            bert_labels.append(result['label'])
        except:
            bert_scores.append(0.0)
            bert_labels.append('NEUTRAL')
    
    df['Sentiment'] = bert_scores
    df['Sentiment_Label'] = bert_labels
    
    print(f"\nSentiment Score Statistics:")
    print(f"  Mean: {df['Sentiment'].mean():.4f}")
    print(f"  Median: {df['Sentiment'].median():.4f}")
    print(f"  Min: {df['Sentiment'].min():.4f}")
    print(f"  Max: {df['Sentiment'].max():.4f}")
    
    print(f"\nSentiment Distribution:")
    print(df['Sentiment_Label'].value_counts())
    
    # CORRELATION WITH IMDB RATING AND GROSS
    # ========================================================================
    print("\n\n2. CORRELATION ANALYSIS")
    print("-" * 60)
    
    # Clean gross data
    df['Gross_Clean'] = df['Gross'].str.replace(',', '').astype(float)
    
    # Calculate correlations
    corr_imdb = df['Sentiment'].corr(df['IMDB_Rating'])
    corr_gross = df['Sentiment'].corr(df['Gross_Clean'])
    
    print(f"\nCorrelation Results:")
    print(f"  Sentiment vs IMDB Rating: {corr_imdb:.4f}")
    print(f"  Sentiment vs Box Office Gross: {corr_gross:.4f}")
    

    # VISUALIZATIONS
    # ========================================================================
    print("\n\nCREATING VISUALIZATIONS")
    print("-" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Sentiment Analysis with BERT', fontsize=16, fontweight='bold')
    
    # Plot 1: Sentiment Distribution
    axes[0].hist(df['Sentiment'], bins=30, color='coral', edgecolor='black')
    axes[0].axvline(df['Sentiment'].mean(), color='red', 
                    linestyle='--', label=f"Mean: {df['Sentiment'].mean():.2f}")
    axes[0].set_xlabel('Sentiment Score')
    axes[0].set_ylabel('Number of Movies')
    axes[0].set_title('BERT Sentiment Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Sentiment vs IMDB Rating
    axes[1].scatter(df['Sentiment'], df['IMDB_Rating'], alpha=0.5, color='orange')
    z = np.polyfit(df['Sentiment'], df['IMDB_Rating'], 1)
    p = np.poly1d(z)
    axes[1].plot(df['Sentiment'], p(df['Sentiment']), "r--", alpha=0.8)
    axes[1].set_xlabel('Sentiment Score')
    axes[1].set_ylabel('IMDB Rating')
    axes[1].set_title(f'Sentiment vs IMDB Rating (r = {corr_imdb:.3f})')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Sentiment vs Gross
    df_gross = df.dropna(subset=['Gross_Clean'])
    axes[2].scatter(df_gross['Sentiment'], df_gross['Gross_Clean']/1e6, 
                   alpha=0.5, color='brown')
    axes[2].set_xlabel('Sentiment Score')
    axes[2].set_ylabel('Box Office Gross (Millions $)')
    axes[2].set_title(f'Sentiment vs Gross (r = {corr_gross:.3f})')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bert_sentiment_correlation.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'bert_sentiment_correlation.png'")
    plt.show()
    
    print("\nAnalysis Complete!")
    return df



def predict_imdb_rating(df_or_path):
    """
    Predicts IMDB ratings using text features, genre, runtime, certificate, and votes.
    Compares Linear Regression, Random Forest, XGBoost, and Deep Learning.
    
    Args:
        df_or_path: Either a preprocessed DataFrame or path to CSV file
    """
    
    # Load dataset if path provided, otherwise use DataFrame
    if isinstance(df_or_path, str):
        df = load_and_preprocess(df_or_path)
    else:
        df = df_or_path.copy()
    
    print("IMDB RATING PREDICTION")
    print("=" * 70)
    
    # FEATURE ENGINEERING
    # ========================================================================
    print("\n1. FEATURE ENGINEERING")
    print("-" * 70)
    
    # Drop rows with missing target or important features
    df = df.dropna(subset=['IMDB_Rating', 'Overview', 'Runtime_Minutes', 'No_of_Votes'])
    
    print(f"Dataset size: {len(df)} movies")
    
    # Runtime_Minutes already created by load_and_preprocess()
    # No need to recreate it
    
    # Encode Certificate (G, PG, R, etc.)
    le_cert = LabelEncoder()
    df['Certificate_Encoded'] = le_cert.fit_transform(df['Certificate'].fillna('Unknown'))
    
    # Extract overview embeddings using TF-IDF
    print("\nExtracting text features from overviews...")
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    overview_features = tfidf.fit_transform(df['Overview']).toarray()
    
    # Create genre features (one-hot encoding for top genres)
    print("Processing genre features...")
    all_genres = []
    for genres in df['Genre'].dropna():
        all_genres.extend([g.strip() for g in genres.split(',')])
    
    from collections import Counter
    top_genres = [g[0] for g in Counter(all_genres).most_common(10)]
    
    for genre in top_genres:
        df[f'Genre_{genre}'] = df['Genre'].apply(
            lambda x: 1 if genre in str(x) else 0
        )
    
    # Combine all features
    print("Combining all features...")
    
    # Numerical features
    numerical_features = df[['Runtime_Minutes', 'Certificate_Encoded', 'No_of_Votes']].values
    
    # Genre features
    genre_features = df[[f'Genre_{g}' for g in top_genres]].values
    
    # Combine: Overview embeddings + Numerical + Genre
    X = np.hstack([overview_features, numerical_features, genre_features])
    y = df['IMDB_Rating'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {X.shape[1]} total")
    print(f"  - Overview embeddings: 100")
    print(f"  - Runtime: 1")
    print(f"  - Certificate: 1")
    print(f"  - Votes: 1")
    print(f"  - Genre features: {len(top_genres)}")
    
    # TRAIN-TEST SPLIT
    # ========================================================================
    print("\n\n2. SPLITTING DATA")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} movies")
    print(f"Test set: {len(X_test)} movies")
    
    # TRAIN MODELS
    # ========================================================================
    print("\n\n3. TRAINING MODELS")
    print("-" * 70)
    
    results = {}
    
    # Model 1: Linear Regression
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    results['Linear Regression'] = {'MAE': mae_lr, 'RMSE': rmse_lr}
    
    print(f"  MAE: {mae_lr:.4f}")
    print(f"  RMSE: {rmse_lr:.4f}")
    
    # Model 2: Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    results['Random Forest'] = {'MAE': mae_rf, 'RMSE': rmse_rf}
    
    print(f"  MAE: {mae_rf:.4f}")
    print(f"  RMSE: {rmse_rf:.4f}")
    
    # Model 3: XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    results['XGBoost'] = {'MAE': mae_xgb, 'RMSE': rmse_xgb}
    
    print(f"  MAE: {mae_xgb:.4f}")
    print(f"  RMSE: {rmse_xgb:.4f}")
    
    # Model 4: Deep Learning (Neural Network)
    print("\nTraining Deep Learning (Neural Network)...")
    nn = MLPRegressor(hidden_layer_sizes=(128, 64, 32), 
                      max_iter=500, 
                      random_state=42,
                      early_stopping=True)
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    results['Deep Learning'] = {'MAE': mae_nn, 'RMSE': rmse_nn}
    
    print(f"  MAE: {mae_nn:.4f}")
    print(f"  RMSE: {rmse_nn:.4f}")
    
    # COMPARISON
    # ========================================================================
    print("\n\n4. MODEL COMPARISON")
    print("-" * 70)
    
    print(f"\n{'Model':<20} {'MAE':<12} {'RMSE':<12}")
    print("-" * 50)
    for model, metrics in results.items():
        print(f"{model:<20} {metrics['MAE']:<12.4f} {metrics['RMSE']:<12.4f}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['MAE'])
    print(f"\nBest Model: {best_model[0]} (Lowest MAE: {best_model[1]['MAE']:.4f})")
    
    #  VISUALIZATIONS
    # ========================================================================
    print("\n\n5. CREATING VISUALIZATIONS")
    print("-" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('IMDB Rating Prediction - Model Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: MAE Comparison
    models = list(results.keys())
    maes = [results[m]['MAE'] for m in models]
    
    axes[0, 0].bar(models, maes, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    axes[0, 0].set_ylabel('Mean Absolute Error (MAE)')
    axes[0, 0].set_title('MAE Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: RMSE Comparison
    rmses = [results[m]['RMSE'] for m in models]
    
    axes[0, 1].bar(models, rmses, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    axes[0, 1].set_ylabel('Root Mean Squared Error (RMSE)')
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Linear Regression Predictions
    axes[0, 2].scatter(y_test, y_pred_lr, alpha=0.5)
    axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 2].set_xlabel('Actual Rating')
    axes[0, 2].set_ylabel('Predicted Rating')
    axes[0, 2].set_title(f'Linear Regression (MAE: {mae_lr:.3f})')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Random Forest Predictions
    axes[1, 0].scatter(y_test, y_pred_rf, alpha=0.5, color='green')
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Rating')
    axes[1, 0].set_ylabel('Predicted Rating')
    axes[1, 0].set_title(f'Random Forest (MAE: {mae_rf:.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: XGBoost Predictions
    axes[1, 1].scatter(y_test, y_pred_xgb, alpha=0.5, color='orange')
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual Rating')
    axes[1, 1].set_ylabel('Predicted Rating')
    axes[1, 1].set_title(f'XGBoost (MAE: {mae_xgb:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Deep Learning Predictions
    axes[1, 2].scatter(y_test, y_pred_nn, alpha=0.5, color='red')
    axes[1, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 2].set_xlabel('Actual Rating')
    axes[1, 2].set_ylabel('Predicted Rating')
    axes[1, 2].set_title(f'Deep Learning (MAE: {mae_nn:.3f})')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rating_prediction_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'rating_prediction_comparison.png'")
    plt.show()
    
    return results

def classify_movie_success(csv_file):
    """
    Classifies movies as Hit or Flop based on IMDB rating >= 7.5
    """
    
    df = pd.read_csv(csv_file)
    
    print("MOVIE SUCCESS CLASSIFICATION")
    print("=" * 60)

    # 1. DEFINE SUCCESS LABEL (IMDB >= 8.0 = HIT)
    # ========================================================================
    print("\n1. SUCCESS LABEL DEFINITION")
    print("-" * 60)
    
    df = df.dropna(subset=['IMDB_Rating', 'Overview', 'Runtime', 'No_of_Votes'])
    
    df['Success'] = (df['IMDB_Rating'] >= 8.0).astype(int)
    
    print("Success = IMDB Rating >= 8.0 (Hit)")
    print("Failure = IMDB Rating < 8.0 (Flop)")
    print(f"\nDistribution:")
    print(f"  Hit (1): {(df['Success'] == 1).sum()} movies")
    print(f"  Flop (0): {(df['Success'] == 0).sum()} movies")

    # 2. FEATURE ENGINEERING
    # ========================================================================
    print("\n\n2. PREPARING FEATURES")
    print("-" * 60)
    
    # Text features
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    text_features = tfidf.fit_transform(df['Overview']).toarray()
    
    # Numerical features
    df['Runtime_Minutes'] = df['Runtime'].str.replace(' min', '').astype(float)
    df['Certificate_Encoded'] = LabelEncoder().fit_transform(df['Certificate'].fillna('Unknown'))
    
    X = np.hstack([text_features, 
                   df[['Runtime_Minutes', 'Certificate_Encoded', 'No_of_Votes']].values])
    y = df['Success'].values
    
    print(f"Total features: {X.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    

        # 3. TRAIN CLASSIFIERS
    # ========================================================================
    print("\n\n3. TRAINING CLASSIFIERS")
    print("-" * 60)
    
    results = {}
    
    # Logistic Regression
    print("\nLogistic Regression:")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    results['Logistic Regression'] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred)
    }
    
    for metric, value in results['Logistic Regression'].items():
        print(f"  {metric}: {value:.4f}")
    
    




# Run the function    

if __name__ == '__main__':
    csv_path = 'imdb_top_1000.csv'
    
    # Run all analyses in sequence (using preprocessed data)
    print("Starting IMDB Analysis Pipeline...\n")
    
    # 1. EDA and Genre Analysis
   # df = exploratory_data_analysis(csv_path, plot=True)
   # genre_analysis(csv_path)
    
    # 2. Text Analysis
   # df_processed = overview_text_preprocessing(df)
   # keyword_extraction_tfidf(df_processed)
    
    # 3. Sentiment Analysis (VADER is faster, BERT slow - take time)
   # df_with_sentiment = sentiment_analysis_vader(df_processed)
   # df_with_bert = sentiment_analysis_bert(df_with_sentiment)  # Deep analysis with BERT
    
    # 4. Rating Prediction (reuse preprocessed data)
   # results = predict_imdb_rating(df_processed)  # Pass DataFrame instead of reloading CSV

    # 5. Movie Success Classification
    classify_movie_success('imdb_top_1000.csv')
    
    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETED SUCCESSFULLY!")
    print("=" * 70)




