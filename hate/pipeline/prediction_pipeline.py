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
            self.base_dir = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    )
                )
            )

            logging.info(
                f"Base directory: {self.base_dir}"
            )

            # ==========================================================
            # FIND MODEL
            # ==========================================================
            artifacts_dir = os.path.join(
                self.base_dir,
                "artifacts"
            )

            self.model_path = None

            # Find .h5 / .keras model
            for root, dirs, files in os.walk(artifacts_dir):

                for file in files:

                    if file.endswith(".h5") or file.endswith(".keras"):

                        self.model_path = os.path.join(
                            root,
                            file
                        )

                        break

                if self.model_path:
                    break

            if self.model_path is None:

                raise FileNotFoundError(
                    "No trained model (.h5 or .keras) "
                    "found inside artifacts folder."
                )

            logging.info(
                f"Model found at: {self.model_path}"
            )

            # ==========================================================
            # TOKENIZER
            # ==========================================================
            self.tokenizer_path = os.path.join(
                self.base_dir,
                "tokenizer.pickle"
            )

            if not os.path.exists(
                self.tokenizer_path
            ):

                raise FileNotFoundError(
                    f"Tokenizer not found at: "
                    f"{self.tokenizer_path}"
                )

            logging.info(
                f"Tokenizer found at: "
                f"{self.tokenizer_path}"
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
            # NLTK STOPWORDS
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
                f"Prediction pipeline initialization failed: {str(e)}"
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

            return " ".join(words)

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

            # Clean text
            cleaned_text = self.clean_text(text)

            logging.info(
                f"Cleaned text: {cleaned_text}"
            )

            # Convert text to sequence
            sequence = self.tokenizer.texts_to_sequences(
                [cleaned_text]
            )

            # Padding
            padded = pad_sequences(
                sequence,
                maxlen=MAX_LEN,
                padding="pre",
                truncating="pre"
            )

            # Prediction
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

            # Classification
            if score >= 0.5:

                return {
                    "prediction": "hate and abusive",
                    "score": round(score, 4)
                }

            else:

                return {
                    "prediction": "no hate",
                    "score": round(score, 4)
                }

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

        return self.predict(text)