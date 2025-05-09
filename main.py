from utils.preprocessing import load_and_preprocess_data
from utils.eda import (
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_target_distribution
)
from utils.validation import validate_split

def main():

    csv_path = 'dataset/cardio_data_processed.csv'

    try:
        X_train, X_test, y_train, y_test, df_eda = load_and_preprocess_data(csv_path)
        print("Data Preprocessing Complete!")
        print("Training shape:", X_train.shape)
        print("Testing shape:", X_test.shape)

        validate_split(y_train, y_test)
        num_features = ['Age (Years)', 'Height (cm)', 'Weight (lbs)', 'Body Mass Index (BMI)', 'Systolic Blood Pressure', 'Diastolic Blood Pressure']
        cat_features = ['Gender', 'Cholesterol Level', 'Glucose Level', 'Smoking Status', 'Alcohol Intake', 'Physical Activity']

        plot_numerical_distributions(df_eda, num_features)
        plot_categorical_distributions(df_eda, cat_features)
        plot_target_distribution(df_eda)

    except Exception as e:
        print("Error during preprocessing:", e)

if __name__ == "__main__":
    main()

