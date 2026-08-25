import os
import sys
import re
import string
import pickle
import nltk

from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from hate.logger import logging
from hate.exception import CustomException
from hate.constants import MAX_LEN


class PredictionPipeline:

    def __init__(self):
        try:
            # ==========================================================
            # BASE DIRECTORY
            # ==========================================================
            BASE_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    )
                )
            )

            self.base_dir = BASE_DIR

            logging.info(f"Base directory: {BASE_DIR}")

            # ==========================================================
            # MODEL PATH
            # ==========================================================
            #
            # Keep this pointing to the model that you want to deploy.
            #
            self.model_path = os.path.join(
                BASE_DIR,
                "artifacts",
                "08_26_2026_00_05_56",
                "ModelTrainerArtifacts",
                "model.h5"
            )

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Trained model not found at: {self.model_path}"
                )

            logging.info(
                f"Trained model found at: {self.model_path}"
            )

            # ==========================================================
            # TOKENIZER PATH
            # ==========================================================

            self.tokenizer_path = os.path.join(
                BASE_DIR,
                "tokenizer.pickle"
            )

            if not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError(
                    f"Tokenizer not found at: {self.tokenizer_path}"
                )

            logging.info(
                f"Tokenizer found at: {self.tokenizer_path}"
            )

            # ==========================================================
            # LOAD TOKENIZER
            # ==========================================================

            with open(
                self.tokenizer_path,
                "rb"
            ) as file:

                self.tokenizer = pickle.load(file)

            logging.info(
                "Tokenizer loaded successfully."
            )

            # ==========================================================
            # LOAD NLTK STOPWORDS
            # ==========================================================

            try:
                self.stop_words = set(
                    stopwords.words("english")
                )

            except LookupError:

                nltk.download(
                    "stopwords",
                    quiet=True
                )

                self.stop_words = set(
                    stopwords.words("english")
                )

            # ==========================================================
            # STEMMER
            # ==========================================================

            self.stemmer = nltk.SnowballStemmer(
                "english"
            )

            # ==========================================================
            # LOAD MODEL
            # ==========================================================

            logging.info(
                "Loading TensorFlow model..."
            )

            self.model = keras.models.load_model(
                self.model_path,
                compile=False
            )

            logging.info(
                "TensorFlow model loaded successfully."
            )

        except Exception as e:

            logging.error(
                f"Error initializing PredictionPipeline: {str(e)}"
            )

            raise CustomException(
                e,
                sys
            ) from e

    # ==============================================================
    # TEXT CLEANING
    # ==============================================================

    def clean_text(self, text):

        try:

            # Convert to string
            text = str(text)

            # Lowercase
            text = text.lower()

            # Remove square brackets
            text = re.sub(
                r"\[.*?\]",
                "",
                text
            )

            # Remove URLs
            text = re.sub(
                r"https?://\S+|www\.\S+",
                "",
                text
            )

            # Remove HTML
            text = re.sub(
                r"<.*?>",
                "",
                text
            )

            # Remove punctuation
            text = re.sub(
                "[%s]" % re.escape(string.punctuation),
                "",
                text
            )

            # Remove newline
            text = re.sub(
                r"\n",
                " ",
                text
            )

            # Remove words containing numbers
            text = re.sub(
                r"\w*\d\w*",
                "",
                text
            )

            # Remove extra spaces
            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            # Stopword removal + stemming
            words = []

            for word in text.split():

                if word not in self.stop_words:

                    words.append(
                        self.stemmer.stem(word)
                    )

            cleaned_text = " ".join(words)

            return cleaned_text

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ==============================================================
    # PREDICTION
    # ==============================================================

    def predict(self, text):

        try:

            logging.info(
                "Running hate speech prediction..."
            )

            # ------------------------------------------------------
            # Clean input text
            # ------------------------------------------------------

            cleaned_text = self.clean_text(text)

            logging.info(
                f"Cleaned text: {cleaned_text}"
            )

            # ------------------------------------------------------
            # Convert text into sequence
            # ------------------------------------------------------

            sequence = self.tokenizer.texts_to_sequences(
                [cleaned_text]
            )

            # ------------------------------------------------------
            # Padding
            # ------------------------------------------------------

            padded_sequence = pad_sequences(
                sequence,
                maxlen=MAX_LEN,
                padding="pre",
                truncating="pre"
            )

            # ------------------------------------------------------
            # Model prediction
            # ------------------------------------------------------

            prediction = self.model.predict(
                padded_sequence,
                verbose=0
            )

            score = float(
                prediction[0][0]
            )

            logging.info(
                f"Prediction score: {score}"
            )

            # ------------------------------------------------------
            # Classification
            # ------------------------------------------------------

            if score >= 0.5:

                result = {
                    "prediction": "hate and abusive",
                    "score": round(score, 4)
                }

            else:

                result = {
                    "prediction": "no hate",
                    "score": round(score, 4)
                }

            logging.info(
                f"Prediction result: {result}"
            )

            return result

        except Exception as e:

            logging.error(
                f"Prediction failed: {str(e)}"
            )

            raise CustomException(
                e,
                sys
            ) from e

    # ==============================================================
    # PIPELINE
    # ==============================================================

    def run_pipeline(self, text):

        try:

            return self.predict(text)

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e