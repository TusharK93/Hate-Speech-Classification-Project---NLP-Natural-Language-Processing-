import os
import sys
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from hate.logger import logging
from hate.exception import CustomException
from hate.constants import *

from hate.ml.model import ModelArchitecture

from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import (
    ModelTrainerArtifacts,
    DataTransformationArtifacts
)


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifacts: DataTransformationArtifacts,
        model_trainer_config: ModelTrainerConfig
    ):

        self.data_transformation_artifacts = (
            data_transformation_artifacts
        )

        self.model_trainer_config = (
            model_trainer_config
        )

    # ==========================================================
    # SPLIT DATA
    # ==========================================================

    def splitting_data(self, csv_path):

        try:

            logging.info(
                "Reading transformed dataset"
            )

            df = pd.read_csv(csv_path)

            # Keep required columns
            df = df[
                [TWEET, LABEL]
            ]

            # Remove missing values
            df = df.dropna(
                subset=[TWEET, LABEL]
            )

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
                stratify=y
            )

            return (
                x_train,
                x_test,
                y_train,
                y_test
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ==========================================================
    # TOKENIZATION
    # ==========================================================

    def tokenizing(self, x_train):

        try:

            logging.info(
                "Tokenizing data"
            )

            x_train = (
                x_train
                .fillna("")
                .astype(str)
            )

            tokenizer = Tokenizer(
                num_words=self.model_trainer_config.MAX_WORDS
            )

            tokenizer.fit_on_texts(
                x_train
            )

            sequences = tokenizer.texts_to_sequences(
                x_train
            )

            sequences_matrix = pad_sequences(
                sequences,
                maxlen=self.model_trainer_config.MAX_LEN
            )

            return (
                sequences_matrix,
                tokenizer
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ==========================================================
    # MODEL TRAINING
    # ==========================================================

    def initiate_model_trainer(self):

        try:

            logging.info(
                "Starting Model Trainer"
            )

            # --------------------------------------------------
            # SPLIT DATA
            # --------------------------------------------------

            (
                x_train,
                x_test,
                y_train,
                y_test
            ) = self.splitting_data(
                self.data_transformation_artifacts
                .transformed_data_path
            )

            # --------------------------------------------------
            # CREATE MODEL
            # --------------------------------------------------

            model = ModelArchitecture().get_model()

            # --------------------------------------------------
            # TOKENIZATION
            # --------------------------------------------------

            (
                sequences_matrix,
                tokenizer
            ) = self.tokenizing(
                x_train
            )

            # --------------------------------------------------
            # TRAIN MODEL
            # --------------------------------------------------

            model.fit(
                sequences_matrix,
                y_train,
                batch_size=self.model_trainer_config.BATCH_SIZE,
                epochs=self.model_trainer_config.EPOCH,
                validation_split=self.model_trainer_config.VALIDATION_SPLIT,
                verbose=1
            )

            # --------------------------------------------------
            # CREATE MODEL DIRECTORY
            # --------------------------------------------------

            os.makedirs(
                self.model_trainer_config.TRAINED_MODEL_DIR,
                exist_ok=True
            )

            # --------------------------------------------------
            # SAVE MODEL
            # --------------------------------------------------

            model.save(
                self.model_trainer_config.TRAINED_MODEL_PATH
            )

            logging.info(
                f"Model saved at: "
                f"{self.model_trainer_config.TRAINED_MODEL_PATH}"
            )

            # --------------------------------------------------
            # SAVE TOKENIZER
            # --------------------------------------------------

            tokenizer_path = os.path.join(
                self.model_trainer_config.TRAINED_MODEL_DIR,
                "tokenizer.pickle"
            )

            with open(
                tokenizer_path,
                "wb"
            ) as file:

                pickle.dump(
                    tokenizer,
                    file
                )

            # Also save tokenizer at project root
            # so prediction pipeline can use it.

            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    )
                )
            )

            root_tokenizer_path = os.path.join(
                project_root,
                "tokenizer.pickle"
            )

            with open(
                root_tokenizer_path,
                "wb"
            ) as file:

                pickle.dump(
                    tokenizer,
                    file
                )

            logging.info(
                f"Tokenizer saved at: "
                f"{root_tokenizer_path}"
            )

            # --------------------------------------------------
            # SAVE TEST DATA
            # --------------------------------------------------

            x_test_df = pd.DataFrame({
                TWEET: x_test.values
            })

            y_test_df = pd.DataFrame({
                LABEL: y_test.values
            })

            x_test_df.to_csv(
                self.model_trainer_config.X_TEST_DATA_PATH,
                index=False
            )

            y_test_df.to_csv(
                self.model_trainer_config.Y_TEST_DATA_PATH,
                index=False
            )

            # --------------------------------------------------
            # SAVE TRAIN DATA
            # --------------------------------------------------

            x_train_df = pd.DataFrame({
                TWEET: x_train.values
            })

            x_train_df.to_csv(
                self.model_trainer_config.X_TRAIN_DATA_PATH,
                index=False
            )

            logging.info(
                "Model training completed successfully."
            )

            # --------------------------------------------------
            # ARTIFACT
            # --------------------------------------------------

            model_trainer_artifacts = ModelTrainerArtifacts(

                trained_model_path=(
                    self.model_trainer_config
                    .TRAINED_MODEL_PATH
                ),

                x_test_path=(
                    self.model_trainer_config
                    .X_TEST_DATA_PATH
                ),

                y_test_path=(
                    self.model_trainer_config
                    .Y_TEST_DATA_PATH
                )
            )

            return model_trainer_artifacts

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e