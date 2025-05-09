from utils.preprocessing import load_and_preprocess_data
from models.training_models import ModelTrainer

def main():
    csv_path = "dataset/cardio_data_processed.csv"
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(csv_path)

    trainer = ModelTrainer(X_train, X_test, y_train, y_test)

    trainer.train_all_models()

    print("All models trained and saved!")

if __name__ == "__main__":
    main()
