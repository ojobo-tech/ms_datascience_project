from utils.preprocessing import load_and_preprocess_data
from utils.eda import (
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_target_distribution
)
from utils.validation import validate_split

def main():
    print("Running main.py")
    print("Starting Data Preprocessing...")

    csv_path = 'dataset/cardio_data_processed.csv'

    try:
        print("Loading and preprocessing...")
        X_train, X_test, y_train, y_test, df_eda = load_and_preprocess_data(csv_path)
        print("Data Preprocessing Complete!")
        print("Training shape:", X_train.shape)
        print("Testing shape:", X_test.shape)

        validate_split(y_train, y_test)

        # Define features for EDA
        num_features = ['age_years', 'height', 'weight_lb', 'bmi', 'ap_hi', 'ap_lo']
        cat_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

        print("\nRunning Exploratory Data Analysis...")
        plot_numerical_distributions(df_eda, num_features)
        plot_categorical_distributions(df_eda, cat_features)
        plot_target_distribution(df_eda)

    except Exception as e:
        print("Error during preprocessing:", e)

if __name__ == "__main__":
    main()

