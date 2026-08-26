import os
import sys
import re
import string
import json

import nltk
import numpy as np

from nltk.corpus import stopwords

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from hate.logger import logging
from hate.exception import CustomException
from hate.constants import MAX_LEN


class PredictionPipeline:

    def __init__(self):

        try:

            # ==================================================
            # BASE DIRECTORY
            # ==================================================

            self.base_dir = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    )
                )
            )

            logging.info(
                f"Project base directory: {self.base_dir}"
            )


            # ==================================================
            # ARTIFACTS DIRECTORY
            # ==================================================

            self.artifacts_dir = os.path.join(
                self.base_dir,
                "artifacts"
            )

            logging.info(
                f"Artifacts directory: {self.artifacts_dir}"
            )


            # ==================================================
            # FIND MODEL
            # ==================================================

            self.model_path = self._find_model()

            logging.info(
                f"Selected model: {self.model_path}"
            )


            # ==================================================
            # TOKENIZER JSON
            # ==================================================

            self.tokenizer_path = os.path.join(
                self.base_dir,
                "tokenizer.json"
            )

            if not os.path.exists(self.tokenizer_path):

                raise FileNotFoundError(
                    f"tokenizer.json not found at: "
                    f"{self.tokenizer_path}"
                )

            logging.info(
                f"Tokenizer found: {self.tokenizer_path}"
            )


            # ==================================================
            # LOAD TOKENIZER
            # ==================================================

            self.tokenizer = self._load_tokenizer()

            logging.info(
                "Tokenizer loaded successfully."
            )


            # ==================================================
            # STOPWORDS
            # ==================================================

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


            # ==================================================
            # STEMMER
            # ==================================================

            self.stemmer = nltk.SnowballStemmer(
                "english"
            )


            # ==================================================
            # LOAD MODEL
            # ==================================================

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


            # ==================================================
            # MODEL STATUS
            # ==================================================

            self.model_loaded = True

            logging.info(
                "Prediction pipeline initialized successfully."
            )


        except Exception as e:

            self.model = None
            self.tokenizer = None
            self.model_loaded = False

            logging.exception(
                "Prediction pipeline initialization failed."
            )

            # Do NOT crash FastAPI completely.
            # This allows /health to report the real problem.

            self.initialization_error = str(e)


    # ==========================================================
    # FIND MODEL
    # ==========================================================

    def _find_model(self):

        if not os.path.exists(self.artifacts_dir):

            raise FileNotFoundError(
                f"Artifacts directory not found: "
                f"{self.artifacts_dir}"
            )

        model_files = []

        for root, dirs, files in os.walk(
            self.artifacts_dir
        ):

            for file in files:

                if file.endswith(".h5"):

                    model_files.append(
                        os.path.join(
                            root,
                            file
                        )
                    )

                elif file.endswith(".keras"):

                    model_files.append(
                        os.path.join(
                            root,
                            file
                        )
                    )


        if not model_files:

            raise FileNotFoundError(
                "No .h5 or .keras model found "
                "inside artifacts directory."
            )


        # ======================================================
        # Select newest model
        # ======================================================

        model_files.sort(
            key=os.path.getmtime,
            reverse=True
        )

        return model_files[0]


    # ==========================================================
    # LOAD TOKENIZER FROM JSON
    # ==========================================================

    def _load_tokenizer(self):

        with open(
            self.tokenizer_path,
            "r",
            encoding="utf-8"
        ) as f:

            tokenizer_config = json.load(f)


        # ======================================================
        # Keras tokenizer JSON structure
        # ======================================================

        config = tokenizer_config["config"]


        # Keras JSON values can be encoded strings
        def decode_value(value):

            if isinstance(value, str):

                try:

                    return json.loads(value)

                except Exception:

                    return value

            return value


        word_counts = decode_value(
            config.get("word_counts", "{}")
        )

        word_docs = decode_value(
            config.get("word_docs", "{}")
        )

        index_docs = decode_value(
            config.get("index_docs", "{}")
        )

        index_word = decode_value(
            config.get("index_word", "{}")
        )

        word_index = decode_value(
            config.get("word_index", "{}")
        )


        # ======================================================
        # Normalize dictionary keys
        # ======================================================

        index_word = {
            int(k): v
            for k, v in index_word.items()
        }

        index_docs = {
            int(k): int(v)
            for k, v in index_docs.items()
        }


        # ======================================================
        # Custom tokenizer object
        # ======================================================

        class JSONTokenizer:

            def __init__(self):

                self.word_index = word_index
                self.index_word = index_word

                self.num_words = config.get(
                    "num_words",
                    None
                )

                if self.num_words is not None:

                    try:

                        self.num_words = int(
                            self.num_words
                        )

                    except Exception:

                        pass

                self.oov_token = config.get(
                    "oov_token",
                    None
                )

                self.lower = config.get(
                    "lower",
                    True
                )

                self.split = config.get(
                    "split",
                    " "
                )


            def texts_to_sequences(
                self,
                texts
            ):

                sequences = []


                for text in texts:

                    if text is None:

                        text = ""

                    text = str(text)


                    if self.lower:

                        text = text.lower()


                    words = text.split(
                        self.split
                    )

                    sequence = []


                    for word in words:

                        if word in self.word_index:

                            index = self.word_index[word]

                        elif self.oov_token is not None:

                            index = self.word_index.get(
                                self.oov_token
                            )

                        else:

                            continue


                        if index is None:

                            continue


                        if (
                            self.num_words is None
                            or index < self.num_words
                        ):

                            sequence.append(
                                int(index)
                            )


                    sequences.append(sequence)


                return sequences


        return JSONTokenizer()


    # ==========================================================
    # TEXT CLEANING
    # ==========================================================

    def clean_text(
        self,
        text
    ):

        text = str(text).lower()


        # ------------------------------------------------------
        # Remove square brackets
        # ------------------------------------------------------

        text = re.sub(
            r"\[.*?\]",
            "",
            text
        )


        # ------------------------------------------------------
        # Remove URLs
        # ------------------------------------------------------

        text = re.sub(
            r"https?://\S+|www\.\S+",
            "",
            text
        )


        # ------------------------------------------------------
        # Remove HTML
        # ------------------------------------------------------

        text = re.sub(
            r"<.*?>",
            "",
            text
        )


        # ------------------------------------------------------
        # Remove punctuation
        # ------------------------------------------------------

        text = re.sub(
            "[%s]" % re.escape(
                string.punctuation
            ),
            "",
            text
        )


        # ------------------------------------------------------
        # Remove newline
        # ------------------------------------------------------

        text = re.sub(
            r"\n",
            "",
            text
        )


        # ------------------------------------------------------
        # Remove words containing numbers
        # ------------------------------------------------------

        text = re.sub(
            r"\w*\d\w*",
            "",
            text
        )


        # ------------------------------------------------------
        # Stopword removal + stemming
        # ------------------------------------------------------

        words = []

        for word in text.split():

            if word not in self.stop_words:

                words.append(
                    self.stemmer.stem(word)
                )


        return " ".join(words)


    # ==========================================================
    # PREDICTION
    # ==========================================================

    def predict(
        self,
        text
    ):

        try:

            if not self.model_loaded:

                return {
                    "status": "error",
                    "message": (
                        "Prediction model is not loaded."
                    )
                }


            # --------------------------------------------------
            # Validate input
            # --------------------------------------------------

            if text is None:

                return {
                    "status": "error",
                    "message": "Text cannot be None."
                }


            text = str(text).strip()


            if not text:

                return {
                    "status": "error",
                    "message": "Text cannot be empty."
                }


            logging.info(
                "Running hate speech prediction..."
            )


            # --------------------------------------------------
            # Clean text
            # --------------------------------------------------

            cleaned_text = self.clean_text(
                text
            )


            logging.info(
                f"Cleaned text: {cleaned_text}"
            )


            # --------------------------------------------------
            # Convert text to sequence
            # --------------------------------------------------

            sequence = (
                self.tokenizer.texts_to_sequences(
                    [cleaned_text]
                )
            )


            # --------------------------------------------------
            # Padding
            # --------------------------------------------------

            padded = pad_sequences(
                sequence,
                maxlen=MAX_LEN,
                padding="pre",
                truncating="pre"
            )


            # --------------------------------------------------
            # Prediction
            # --------------------------------------------------

            prediction = self.model.predict(
                padded,
                verbose=0
            )


            score = float(
                np.asarray(
                    prediction
                ).reshape(-1)[0]
            )


            logging.info(
                f"Prediction score: {score}"
            )


            # --------------------------------------------------
            # Classification
            # --------------------------------------------------

            if score >= 0.5:

                label = "hate and abusive"

            else:

                label = "no hate"


            return {
                "status": "success",
                "prediction": label,
                "score": round(
                    score,
                    4
                )
            }


        except Exception as e:

            logging.exception(
                "Prediction failed."
            )

            return {
                "status": "error",
                "message": str(e)
            }


    # ==========================================================
    # PIPELINE
    # ==========================================================

    def run_pipeline(
        self,
        text
    ):

        return self.predict(text)