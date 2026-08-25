import sys

from hate.logger import logging
from hate.exception import CustomException

from hate.components.data_ingestion import DataIngestion
from hate.components.data_transformation import DataTransformation
from hate.components.model_trainer import ModelTrainer
from hate.components.model_evaluation import ModelEvaluation
from hate.components.model_pusher import ModelPusher

from hate.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from hate.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
    ModelPusherArtifacts
)


class TrainPipeline:

    def __init__(self):

        self.data_ingestion_config = (
            DataIngestionConfig()
        )

        self.data_transformation_config = (
            DataTransformationConfig()
        )

        self.model_trainer_config = (
            ModelTrainerConfig()
        )

        self.model_evaluation_config = (
            ModelEvaluationConfig()
        )

        self.model_pusher_config = (
            ModelPusherConfig()
        )

    # ==========================================================
    # DATA INGESTION
    # ==========================================================

    def start_data_ingestion(self):

        try:

            logging.info(
                "Starting data ingestion."
            )

            data_ingestion = DataIngestion(
                data_ingestion_config=(
                    self.data_ingestion_config
                )
            )

            artifacts = (
                data_ingestion
                .initiate_data_ingestion()
            )

            logging.info(
                "Data ingestion completed."
            )

            return artifacts

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ==========================================================
    # DATA TRANSFORMATION
    # ==========================================================

    def start_data_transformation(
        self,
        data_ingestion_artifacts
    ):

        try:

            logging.info(
                "Starting data transformation."
            )

            data_transformation = DataTransformation(
                data_ingestion_artifacts=(
                    data_ingestion_artifacts
                ),
                data_transformation_config=(
                    self.data_transformation_config
                )
            )

            artifacts = (
                data_transformation
                .initiate_data_transformation()
            )

            logging.info(
                "Data transformation completed."
            )

            return artifacts

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ==========================================================
    # MODEL TRAINER
    # ==========================================================

    def start_model_trainer(
        self,
        data_transformation_artifacts
    ):

        try:

            logging.info(
                "Starting model training."
            )

            model_trainer = ModelTrainer(
                data_transformation_artifacts=(
                    data_transformation_artifacts
                ),
                model_trainer_config=(
                    self.model_trainer_config
                )
            )

            artifacts = (
                model_trainer
                .initiate_model_trainer()
            )

            logging.info(
                "Model training completed."
            )

            return artifacts

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ==========================================================
    # MODEL EVALUATION
    # ==========================================================

    def start_model_evaluation(
        self,
        model_trainer_artifacts,
        data_transformation_artifacts
    ):

        try:

            logging.info(
                "Starting model evaluation."
            )

            model_evaluation = ModelEvaluation(
                data_transformation_artifacts=(
                    data_transformation_artifacts
                ),
                model_evaluation_config=(
                    self.model_evaluation_config
                ),
                model_trainer_artifacts=(
                    model_trainer_artifacts
                )
            )

            artifacts = (
                model_evaluation
                .initiate_model_evaluation()
            )

            logging.info(
                "Model evaluation completed."
            )

            return artifacts

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ==========================================================
    # MODEL PUSHER
    # ==========================================================

    def start_model_pusher(self):

        try:

            logging.info(
                "Starting model pusher."
            )

            model_pusher = ModelPusher(
                model_pusher_config=(
                    self.model_pusher_config
                )
            )

            artifacts = (
                model_pusher
                .initiate_model_pusher()
            )

            logging.info(
                "Model pusher completed."
            )

            return artifacts

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ==========================================================
    # COMPLETE TRAINING PIPELINE
    # ==========================================================

    def run_pipeline(self):

        try:

            logging.info(
                "Starting complete training pipeline."
            )

            # 1. Data ingestion
            data_ingestion_artifacts = (
                self.start_data_ingestion()
            )

            # 2. Data transformation
            data_transformation_artifacts = (
                self.start_data_transformation(
                    data_ingestion_artifacts
                )
            )

            # 3. Model training
            model_trainer_artifacts = (
                self.start_model_trainer(
                    data_transformation_artifacts
                )
            )

            # 4. Model evaluation
            model_evaluation_artifacts = (
                self.start_model_evaluation(
                    model_trainer_artifacts,
                    data_transformation_artifacts
                )
            )

            # 5. Check model
            if not model_evaluation_artifacts.is_model_accepted:

                raise Exception(
                    "Trained model is not better "
                    "than the best model."
                )

            # 6. Push model
            model_pusher_artifacts = (
                self.start_model_pusher()
            )

            logging.info(
                "Complete training pipeline finished successfully."
            )

            return model_pusher_artifacts

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e