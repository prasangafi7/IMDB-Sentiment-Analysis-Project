import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Optional


REQUIRED_COLUMNS = [
    'Series_Title', 'Genre', 'IMDB_Rating', 'Runtime', 'Gross', 'Released_Year'
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

    print("\nAnalysis Complete!")


if __name__ == '__main__':
    csv_path = 'imdb_top_1000.csv'
    df = exploratory_data_analysis(csv_path, plot=True)
    genre_analysis(csv_path)