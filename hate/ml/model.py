from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import RMSprop

from hate.constants import (
    MAX_WORDS,
    MAX_LEN,
    ACTIVATION,
    LOSS,
    METRICS
)


class ModelArchitecture:

    def __init__(self):
        pass

    def get_model(self):

        # Lightweight model for low-memory deployment
        model = Sequential([
            Embedding(
                input_dim=MAX_WORDS,
                output_dim=32,
                input_length=MAX_LEN
            ),

            SpatialDropout1D(0.2),

            LSTM(
                32,
                dropout=0.1,
                recurrent_dropout=0.0
            ),

            Dense(
                1,
                activation=ACTIVATION
            )
        ])

        model.compile(
            loss=LOSS,
            optimizer=RMSprop(learning_rate=0.001),
            metrics=METRICS
        )

        model.summary()

        return model