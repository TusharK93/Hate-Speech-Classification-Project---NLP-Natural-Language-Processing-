import sys

from hate.logger import logging
from hate.exception import CustomException

from hate.components.data_ingestion import DataIngestion
from hate.components.data_transformation import DataTransformation
from hate.components.model_trainer import ModelTrainer
from hate.components.model_evaluation import ModelEvaluation

from hate.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)

from hate.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
)


class TrainPipeline:

    def __init__(self):

        # -------------------------------------------------
        # Configuration
        # -------------------------------------------------

        self.data_ingestion_config = DataIngestionConfig()

        self.data_transformation_config = (
            DataTransformationConfig()
        )

        self.model_trainer_config = ModelTrainerConfig()

        self.model_evaluation_config = ModelEvaluationConfig()

    # =====================================================
    # DATA INGESTION
    # =====================================================

    def start_data_ingestion(self) -> DataIngestionArtifacts:

        logging.info(
            "Entered start_data_ingestion method"
        )

        try:

            logging.info(
                "Getting data from GCloud Storage bucket"
            )

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifacts = (
                data_ingestion.initiate_data_ingestion()
            )

            logging.info(
                "Train and validation data obtained successfully"
            )

            logging.info(
                "Exited start_data_ingestion method"
            )

            return data_ingestion_artifacts

        except Exception as e:

            raise CustomException(e, sys) from e

    # =====================================================
    # DATA TRANSFORMATION
    # =====================================================

    def start_data_transformation(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
    ) -> DataTransformationArtifacts:

        logging.info(
            "Entered start_data_transformation method"
        )

        try:

            data_transformation = DataTransformation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_transformation_config=(
                    self.data_transformation_config
                ),
            )

            data_transformation_artifacts = (
                data_transformation.initiate_data_transformation()
            )

            logging.info(
                "Data transformation completed successfully"
            )

            logging.info(
                "Exited start_data_transformation method"
            )

            return data_transformation_artifacts

        except Exception as e:

            raise CustomException(e, sys) from e

    # =====================================================
    # MODEL TRAINING
    # =====================================================

    def start_model_trainer(
        self,
        data_transformation_artifacts: DataTransformationArtifacts,
    ) -> ModelTrainerArtifacts:

        logging.info(
            "Entered start_model_trainer method"
        )

        try:

            model_trainer = ModelTrainer(
                data_transformation_artifacts=(
                    data_transformation_artifacts
                ),
                model_trainer_config=(
                    self.model_trainer_config
                ),
            )

            model_trainer_artifacts = (
                model_trainer.initiate_model_trainer()
            )

            logging.info(
                "Model training completed successfully"
            )

            logging.info(
                "Exited start_model_trainer method"
            )

            return model_trainer_artifacts

        except Exception as e:

            raise CustomException(e, sys) from e

    # =====================================================
    # MODEL EVALUATION
    # =====================================================

    def start_model_evaluation(
        self,
        model_trainer_artifacts: ModelTrainerArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts,
    ) -> ModelEvaluationArtifacts:

        logging.info(
            "Entered start_model_evaluation method"
        )

        try:

            model_evaluation = ModelEvaluation(
                data_transformation_artifacts=(
                    data_transformation_artifacts
                ),
                model_evaluation_config=(
                    self.model_evaluation_config
                ),
                model_trainer_artifacts=(
                    model_trainer_artifacts
                ),
            )

            model_evaluation_artifacts = (
                model_evaluation.initiate_model_evaluation()
            )

            logging.info(
                "Model evaluation completed successfully"
            )

            logging.info(
                "Exited start_model_evaluation method"
            )

            return model_evaluation_artifacts

        except Exception as e:

            raise CustomException(e, sys) from e

    # =====================================================
    # COMPLETE TRAINING PIPELINE
    # =====================================================

    def run_pipeline(self) -> None:

        logging.info(
            "Entered run_pipeline method"
        )

        try:

            # -------------------------------------------------
            # 1. Data Ingestion
            # -------------------------------------------------

            logging.info(
                "STEP 1: Starting data ingestion"
            )

            data_ingestion_artifacts = (
                self.start_data_ingestion()
            )

            logging.info(
                "STEP 1: Data ingestion completed"
            )

            # -------------------------------------------------
            # 2. Data Transformation
            # -------------------------------------------------

            logging.info(
                "STEP 2: Starting data transformation"
            )

            data_transformation_artifacts = (
                self.start_data_transformation(
                    data_ingestion_artifacts=(
                        data_ingestion_artifacts
                    )
                )
            )

            logging.info(
                "STEP 2: Data transformation completed"
            )

            # -------------------------------------------------
            # 3. Model Training
            # -------------------------------------------------

            logging.info(
                "STEP 3: Starting model training"
            )

            model_trainer_artifacts = (
                self.start_model_trainer(
                    data_transformation_artifacts=(
                        data_transformation_artifacts
                    )
                )
            )

            logging.info(
                "STEP 3: Model training completed"
            )

            # -------------------------------------------------
            # 4. Model Evaluation
            # -------------------------------------------------

            logging.info(
                "STEP 4: Starting model evaluation"
            )

            model_evaluation_artifacts = (
                self.start_model_evaluation(
                    model_trainer_artifacts=(
                        model_trainer_artifacts
                    ),
                    data_transformation_artifacts=(
                        data_transformation_artifacts
                    ),
                )
            )

            logging.info(
                "STEP 4: Model evaluation completed"
            )

            # -------------------------------------------------
            # 5. Check Model Acceptance
            # -------------------------------------------------

            if not model_evaluation_artifacts.is_model_accepted:

                logging.error(
                    "Trained model was not accepted"
                )

                raise Exception(
                    "Trained model is not better than "
                    "the best model"
                )

            logging.info(
                "Model accepted successfully"
            )

            # -------------------------------------------------
            # 6. Model Pusher DISABLED
            # -------------------------------------------------

            logging.info(
                "STEP 5: Model Pusher skipped"
            )

            logging.info(
                "Google Cloud Model Pusher is disabled "
                "because the GCloud billing account is closed."
            )

            logging.info(
                "The trained model will be deployed "
                "directly with the application."
            )

            # -------------------------------------------------
            # Pipeline Completed
            # -------------------------------------------------

            logging.info(
                "=========================================="
            )

            logging.info(
                "TRAINING PIPELINE COMPLETED SUCCESSFULLY"
            )

            logging.info(
                "=========================================="
            )

        except Exception as e:

            logging.error(
                "Training pipeline failed"
            )

            raise CustomException(e, sys) from e