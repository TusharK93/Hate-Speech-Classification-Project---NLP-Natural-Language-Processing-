import os
import sys
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException

from hate.ml.model import ModelArchitecture

from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import (
    ModelTrainerArtifacts,
    DataTransformationArtifacts,
)


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifacts: DataTransformationArtifacts,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    # ---------------------------------------------------------
    # Split Data
    # ---------------------------------------------------------
    def spliting_data(self, csv_path):

        try:
            logging.info("Reading transformed dataset")

            df = pd.read_csv(csv_path)

            # Keep only required columns
            df = df[[TWEET, LABEL]]

            # Remove missing values
            df = df.dropna(subset=[TWEET, LABEL])

            # Correct datatype
            df[TWEET] = df[TWEET].astype(str)
            df[LABEL] = df[LABEL].astype(int)

            x = df[TWEET]
            y = df[LABEL]

            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.30,
                random_state=self.model_trainer_config.RANDOM_STATE,
                stratify=y,
            )

            logging.info(
                f"Training samples: {len(x_train)}"
            )

            logging.info(
                f"Testing samples: {len(x_test)}"
            )

            return x_train, x_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys) from e

    # ---------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------
    def tokenizing(self, x_train):

        try:
            logging.info("Starting tokenization")

            x_train = x_train.fillna("").astype(str)

            tokenizer = Tokenizer(
                num_words=self.model_trainer_config.MAX_WORDS,
                oov_token="<OOV>"
            )

            tokenizer.fit_on_texts(x_train)

            sequences = tokenizer.texts_to_sequences(x_train)

            sequences_matrix = pad_sequences(
                sequences,
                maxlen=self.model_trainer_config.MAX_LEN,
                padding="post",
                truncating="post"
            )

            logging.info(
                f"Tokenization completed. Shape: {sequences_matrix.shape}"
            )

            return sequences_matrix, tokenizer

        except Exception as e:
            raise CustomException(e, sys) from e

    # ---------------------------------------------------------
    # Model Training
    # ---------------------------------------------------------
    def initiate_model_trainer(self):

        try:
            logging.info("Starting Model Trainer")

            # -------------------------------------------------
            # Read and split data
            # -------------------------------------------------
            x_train, x_test, y_train, y_test = self.spliting_data(
                self.data_transformation_artifacts.transformed_data_path
            )

            # -------------------------------------------------
            # Tokenize training text
            # -------------------------------------------------
            sequences_matrix, tokenizer = self.tokenizing(
                x_train
            )

            # -------------------------------------------------
            # Create lightweight model
            # -------------------------------------------------
            model = ModelArchitecture().get_model()

            # -------------------------------------------------
            # Train model
            # -------------------------------------------------
            logging.info("Starting model training")

            model.fit(
                sequences_matrix,
                y_train,
                batch_size=16,
                epochs=2,
                validation_split=0.20,
                verbose=1
            )

            logging.info("Model training completed")

            # -------------------------------------------------
            # Create model directory
            # -------------------------------------------------
            os.makedirs(
                self.model_trainer_config.TRAINED_MODEL_DIR,
                exist_ok=True
            )

            # -------------------------------------------------
            # Save trained model
            # -------------------------------------------------
            model.save(
                self.model_trainer_config.TRAINED_MODEL_PATH
            )

            logging.info(
                f"Model saved at: "
                f"{self.model_trainer_config.TRAINED_MODEL_PATH}"
            )

            # -------------------------------------------------
            # Save tokenizer
            # -------------------------------------------------
            project_root = os.getcwd()

            tokenizer_path = os.path.join(
                project_root,
                "tokenizer.pickle"
            )

            with open(tokenizer_path, "wb") as file:
                pickle.dump(tokenizer, file)

            logging.info(
                f"Tokenizer saved at: {tokenizer_path}"
            )

            # -------------------------------------------------
            # Convert Series to DataFrames
            # -------------------------------------------------
            x_train_df = pd.DataFrame({
                TWEET: x_train.values
            })

            x_test_df = pd.DataFrame({
                TWEET: x_test.values
            })

            y_test_df = pd.DataFrame({
                LABEL: y_test.values
            })

            # -------------------------------------------------
            # Save test/train CSV files
            # -------------------------------------------------
            x_train_df.to_csv(
                self.model_trainer_config.X_TRAIN_DATA_PATH,
                index=False
            )

            x_test_df.to_csv(
                self.model_trainer_config.X_TEST_DATA_PATH,
                index=False
            )

            y_test_df.to_csv(
                self.model_trainer_config.Y_TEST_DATA_PATH,
                index=False
            )

            logging.info(
                "Training and testing CSV files saved successfully"
            )

            # -------------------------------------------------
            # Create ModelTrainerArtifacts
            # -------------------------------------------------
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=(
                    self.model_trainer_config.TRAINED_MODEL_PATH
                ),
                x_test_path=(
                    self.model_trainer_config.X_TEST_DATA_PATH
                ),
                y_test_path=(
                    self.model_trainer_config.Y_TEST_DATA_PATH
                ),
            )

            logging.info(
                "ModelTrainerArtifacts created successfully"
            )

            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e