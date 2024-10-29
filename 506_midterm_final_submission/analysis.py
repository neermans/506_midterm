import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
df = pd.read_csv("train.modded.csv")

# Set the style for seaborn
sns.set(style="whitegrid")

# Plot 1: Distribution of Review Scores !
plt.figure(figsize=(8, 6))
sns.countplot(x='Score', data=df, palette='viridis')
plt.title("Distribution of Review Scores")
plt.xlabel("Review Score")
plt.ylabel("Count")
plt.show()

# Plot 2: Distribution of Text Length !
plt.figure(figsize=(10, 6))
sns.histplot(df['TextLength'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Text Length in Reviews")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.show()

# Plot 3: Helpfulness Ratio by Score !
plt.figure(figsize=(10, 6))
sns.boxplot(x='Score', y='Helpfulness', data=df, palette='coolwarm')
plt.title("Helpfulness Ratio by Review Score")
plt.xlabel("Review Score")
plt.ylabel("Helpfulness Ratio")
plt.show()

# Plot 4: Exclamation Count Distribution by Score !
plt.figure(figsize=(10, 6))
sns.boxplot(x='Score', y='ExclamationCount', data=df, palette='magma')
plt.title("Distribution of Exclamation Count by Review Score")
plt.xlabel("Review Score")
plt.ylabel("Exclamation Count")
plt.show()


# Plot 5: Positive Text Percentage by Score !
plt.figure(figsize=(10, 6))
sns.boxplot(x='Score', y='PercPositiveText', data=df, palette='plasma')
plt.title("Positive Text Percentage by Review Score")
plt.xlabel("Review Score")
plt.ylabel("Positive Text Percentage")
plt.show()

