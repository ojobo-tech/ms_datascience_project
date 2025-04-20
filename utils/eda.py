import matplotlib.pyplot as plt
import seaborn as sns

def plot_numerical_distributions(df, num_features):
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(num_features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df, cat_features):
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(cat_features, 1):
        plt.subplot(3, 3, i)
        sns.countplot(x=df[col])
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

def plot_target_distribution(df, target_col='cardio'):
    sns.countplot(x=df[target_col])
    plt.title("Target Variable Distribution")
    plt.show()
    print(df[target_col].value_counts(normalize=True))

