from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    # Step 1: Data Ingestion
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    print("✅ Data Ingestion completed")

    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    print("✅ Data Transformation completed")

    # Step 3: Model Training
    model_trainer = ModelTrainer()
    r2_square = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"✅ Model Training completed with R² score: {r2_square:.3f}")

if __name__ == "__main__":
    main()
