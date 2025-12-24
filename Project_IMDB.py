import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_data_analysis(csv_file, plot=True, save_fig=True, fig_path='eda_results.png'):
    """
    Perform exploratory data analysis on IMDB movie dataset and optionally plot trends over time.
    
    Parameters:
    csv_file (str): Path to the CSV file
    plot (bool): Whether to generate and show/save plots
    save_fig (bool): Whether to save plotted figure to disk when plotting
    fig_path (str): File path for saved figure
    
    Returns:
    pd.DataFrame: Cleaned dataframe with new columns
    """
    
    df = pd.read_csv(csv_file)
    
    print("\n EXPLORATORY DATA ANALYSIS - IMDB TOP 1000 MOVIES")
    print("=" * 60)
    
    # DATASET STATISTICS
    # ============================================================================
    print("\n1. DATASET STATISTICS")
    print("-" * 60)
    
    print(f"Total number of movies: {len(df)}")
    
    df['Runtime_Minutes'] = df['Runtime'].str.replace(' min', '', regex=False).astype(float)
    df['Gross_Earnings'] = df['Gross'].str.replace(',', '', regex=False).astype(float)
    
    # Distribution of IMDB Ratings, Runtimes, and Gross Earnings
    print("\nIMDB Rating Distribution:")
    print(df['IMDB_Rating'].describe())
    
    print("\nRuntime Distribution (in minutes):")
    print(df['Runtime_Minutes'].describe())
    
    print("\nGross Earnings Distribution:")
    print(df['Gross_Earnings'].describe())
    print(f"Note: {df['Gross_Earnings'].isna().sum()} movies have missing gross earnings data")

    # VISUALIZATIONS - TRENDS OVER TIME
    # ============================================================================
    if plot:
        print("\n2. CREATING VISUALIZATIONS FOR TRENDS OVER TIME")
        print("-" * 60)

        # Convert year to numeric
        df['Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('IMDB Movies - Trends Over Time', fontsize=16, fontweight='bold')

        # Plots: Average IMDB Rating by Year and Box Office Gross by Year
        yearly_rating = df.groupby('Year')['IMDB_Rating'].mean()
        axes[0].plot(yearly_rating.index, yearly_rating.values, marker='o', color='green', linewidth=2)
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Average IMDB Rating')
        axes[0].set_title('Average IMDB Rating by Year')
        axes[0].grid(True, alpha=0.3)

        yearly_gross = df.groupby('Year')['Gross_Earnings'].mean() / 1000000  # Convert to millions
        axes[1].plot(yearly_gross.index, yearly_gross.values, marker='o', color='orange', linewidth=2)
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Average Gross (Millions $)')
        axes[1].set_title('Box Office Gross by Year')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_fig:
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualizations saved as '{fig_path}'")
        plt.show()

    # Additional summary statistics
    print("\n3. SUMMARY STATISTICS")
    print("-" * 60)
    print(f"Average IMDB Rating: {df['IMDB_Rating'].mean():.2f}")
    print(f"Average Runtime: {df['Runtime_Minutes'].mean():.1f} minutes")
    print(f"Average Gross Earnings: ${df['Gross_Earnings'].mean()/1000000:.2f} Million")
    # Protect against all-NaN Year
    if df['Year'].dropna().empty:
        print("Year Range: N/A")
    else:
        print(f"Year Range: {int(df['Year'].min())} to {int(df['Year'].max())}")

    return df


# Load and analyze data (plots included by default)
df = exploratory_data_analysis('imdb_top_1000.csv', plot=True)