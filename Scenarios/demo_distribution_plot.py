import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def categorical_balance(df, column, title=None, figsize=(10, 6), cmap='viridis', 
                        autopct='%1.1f%%', startangle=90, explode=None, legend_loc='best'):

    # Get value counts and percentages
    value_counts = df[column].value_counts()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate distinct colors for each category
    color_map = plt.cm.get_cmap(cmap)
    colors = [color_map(i/len(value_counts)) for i in range(len(value_counts))]
    
    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        value_counts, 
        labels=None,  # We'll use a legend instead of labels on the pie
        autopct=autopct,
        startangle=startangle,
        explode=explode,
        colors=colors,
        shadow=False
    )
    
    # Enhance the appearance of percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Add legend
    ax.legend(wedges, value_counts.index, title=column, loc=legend_loc)
    
    # Set title
    if title is None:
        title = f'Distribution of {column}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add count information
    ax.text(
        -0.1, -1.2, 
        f'Total count: {value_counts.sum()}\nClass counts: {", ".join([f"{k}: {v}" for k, v in value_counts.items()])}',
        ha='left'
    )
    
    plt.tight_layout()
    
    # Print imbalance metrics
    print(f"Class distribution for {column}:")
    for category, count in value_counts.items():
        print(f"  {category}: {count} ({count/value_counts.sum()*100:.1f}%)")
    
    return fig, ax

def numerical_balance(df, column, bins=30, figsize=(12, 6), kde=True, title=None, color='skyblue', 
                     edge_color='black', kde_color='darkblue', show_rug=True, stat='count'):

    # Drop missing values for the analysis
    data = df[column].dropna()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the histogram with KDE
    sns.histplot(
        data=data,
        bins=bins,
        kde=kde,
        color=color,
        edgecolor=edge_color,
        line_kws={'color': kde_color, 'lw': 2},
        alpha=0.7,
        ax=ax,
        stat=stat
    )
    
    # Add rugplot if requested
    if show_rug:
        sns.rugplot(data=data, ax=ax, color=kde_color, alpha=0.3)
    
    # Set title
    if title is None:
        title = f'Distribution of {column}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add basic statistics as text
    stats_text = (
        f"Count: {len(data)}\n"
        f"Missing: {df[column].isna().sum()} ({df[column].isna().sum()/len(df)*100:.1f}%)\n"
        f"Min: {data.min():.2f}\n"
        f"Max: {data.max():.2f}\n"
        f"Mean: {data.mean():.2f}\n"
        f"Median: {data.median():.2f}\n"
        f"Std Dev: {data.std():.2f}\n"
        f"Skewness: {data.skew():.2f}\n"
        f"Kurtosis: {data.kurtosis():.2f}"
    )
    
    # Position the text box in the upper right
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Determine outliers (using 1.5 * IQR rule)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    # Print distribution information
    print(f"Distribution analysis for {column}:")
    print(f"  Range: {data.min()} to {data.max()}")
    print(f"  Mean={data.mean():.2f}, Median={data.median():.2f}")

    return fig, ax
