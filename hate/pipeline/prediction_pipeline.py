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
            # =========================================================
            # PROJECT ROOT DIRECTORY
            # =========================================================
            BASE_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    )
                )
            )

            self.base_dir = BASE_DIR

            logging.info(
                f"Project base directory: {BASE_DIR}"
            )

            # =========================================================
            # ARTIFACTS DIRECTORY
            # =========================================================
            artifacts_dir = os.path.join(
                BASE_DIR,
                "artifacts"
            )

            if not os.path.exists(artifacts_dir):
                raise FileNotFoundError(
                    f"Artifacts directory not found: {artifacts_dir}"
                )

            # =========================================================
            # FIND ALL TRAINED MODELS
            # =========================================================
            model_files = []

            for root, dirs, files in os.walk(artifacts_dir):

                for file in files:

                    if file.endswith(".h5") or file.endswith(".keras"):

                        model_files.append(
                            os.path.join(root, file)
                        )

            if not model_files:
                raise FileNotFoundError(
                    "No trained model (.h5 or .keras) "
                    "found inside artifacts directory."
                )

            # =========================================================
            # SELECT LATEST MODEL
            # =========================================================
            self.model_path = max(
                model_files,
                key=os.path.getmtime
            )

            logging.info(
                f"Latest trained model found at: "
                f"{self.model_path}"
            )

            # =========================================================
            # TOKENIZER
            # =========================================================
            self.tokenizer_path = os.path.join(
                BASE_DIR,
                "tokenizer.pickle"
            )

            if not os.path.exists(self.tokenizer_path):

                raise FileNotFoundError(
                    f"tokenizer.pickle not found at: "
                    f"{self.tokenizer_path}"
                )

            logging.info(
                f"Tokenizer found at: {self.tokenizer_path}"
            )

            # =========================================================
            # LOAD TOKENIZER
            # =========================================================
            with open(
                self.tokenizer_path,
                "rb"
            ) as file:

                self.tokenizer = pickle.load(file)

            logging.info(
                "Tokenizer loaded successfully."
            )

            # =========================================================
            # STOPWORDS
            # =========================================================
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

            # =========================================================
            # STEMMER
            # =========================================================
            self.stemmer = nltk.SnowballStemmer(
                "english"
            )

            # =========================================================
            # LOAD MODEL ONCE
            # =========================================================
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
                f"PredictionPipeline initialization failed: {str(e)}"
            )

            raise CustomException(e, sys) from e

    # =============================================================
    # TEXT CLEANING
    # =============================================================

    def clean_text(self, text):

        try:

            text = str(text).lower()

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

            # Stopword removal + stemming
            words = []

            for word in text.split():

                if word not in self.stop_words:

                    words.append(
                        self.stemmer.stem(word)
                    )

            return " ".join(words)

        except Exception as e:

            raise CustomException(e, sys) from e

    # =============================================================
    # PREDICTION
    # =============================================================

    def predict(self, text):

        try:

            logging.info(
                f"Prediction request received: {text}"
            )

            # -----------------------------------------------------
            # Clean text
            # -----------------------------------------------------
            cleaned_text = self.clean_text(text)

            logging.info(
                f"Cleaned text: {cleaned_text}"
            )

            # -----------------------------------------------------
            # Convert text to sequence
            # -----------------------------------------------------
            sequence = self.tokenizer.texts_to_sequences(
                [cleaned_text]
            )

            # -----------------------------------------------------
            # Padding
            # -----------------------------------------------------
            padded = pad_sequences(
                sequence,
                maxlen=MAX_LEN,
                padding="pre",
                truncating="pre"
            )

            # -----------------------------------------------------
            # Prediction
            # -----------------------------------------------------
            prediction = self.model.predict(
                padded,
                verbose=0
            )

            score = float(
                prediction[0][0]
            )

            logging.info(
                f"Prediction score: {score}"
            )

            # Keep probability between 0 and 1
            score = max(
                0.0,
                min(1.0, score)
            )

            # -----------------------------------------------------
            # Classification
            # -----------------------------------------------------
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

            raise CustomException(e, sys) from e

    # =============================================================
    # PIPELINE
    # =============================================================

    def run_pipeline(self, text):

        return self.predict(text)