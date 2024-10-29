import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
df = pd.read_csv("train.modded.csv")

# Set the style for seaborn
sns.set(style="whitegrid")

# Directory to save plots (optional)
# If you want to save plots, uncomment and make sure this directory exists
# import os
# os.makedirs("plots", exist_ok=True)

# Plot 1: Distribution of Review Scores
plt.figure(figsize=(8, 6))
sns.countplot(x='Score', data=df, palette='viridis')
plt.title("Distribution of Review Scores")
plt.xlabel("Review Score")
plt.ylabel("Count")
plt.show()

# Plot 2: Distribution of Text Length
plt.figure(figsize=(10, 6))
sns.histplot(df['TextLength'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Text Length in Reviews")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.show()

# Plot 3: Helpfulness Ratio by Score
plt.figure(figsize=(10, 6))
sns.boxplot(x='Score', y='Helpfulness', data=df, palette='coolwarm')
plt.title("Helpfulness Ratio by Review Score")
plt.xlabel("Review Score")
plt.ylabel("Helpfulness Ratio")
plt.show()

# Plot 4: Exclamation Count Distribution by Score
plt.figure(figsize=(10, 6))
sns.boxplot(x='Score', y='ExclamationCount', data=df, palette='magma')
plt.title("Distribution of Exclamation Count by Review Score")
plt.xlabel("Review Score")
plt.ylabel("Exclamation Count")
plt.show()

# Plot 5: Positive vs. Negative Word Counts in Review Text
plt.figure(figsize=(10, 6))
sns.scatterplot(x='NumPositiveText', y='NumNegativeText', hue='Score', data=df, palette='Spectral', alpha=0.6)
plt.title("Positive vs. Negative Word Counts in Review Text")
plt.xlabel("Positive Word Count")
plt.ylabel("Negative Word Count")
plt.legend(title="Review Score")
plt.show()

# Plot 6: Average Review Score by User
plt.figure(figsize=(10, 6))
sns.histplot(df['AverageReview'], bins=30, kde=True, color='salmon')
plt.title("Distribution of Average Review Score by User")
plt.xlabel("Average Review Score")
plt.ylabel("Frequency")
plt.show()

# Plot 7: Standard Deviation of Review Scores by User
plt.figure(figsize=(10, 6))
sns.histplot(df['StdReview'], bins=30, kde=True, color='dodgerblue')
plt.title("Standard Deviation of Review Scores by User")
plt.xlabel("Standard Deviation of Review Scores")
plt.ylabel("Frequency")
plt.show()

# Plot 8: Positive Text Percentage by Score
plt.figure(figsize=(10, 6))
sns.boxplot(x='Score', y='PercPositiveText', data=df, palette='plasma')
plt.title("Positive Text Percentage by Review Score")
plt.xlabel("Review Score")
plt.ylabel("Positive Text Percentage")
plt.show()

# Plot 9: Correlation Heatmap of All Numerical Features
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# Optional: Save all figures if needed
# for i, fig in enumerate(plt.get_fignums(), start=1):
#     plt.figure(fig)
#     plt.savefig(f"plots/plot_{i}.png")

