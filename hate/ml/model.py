from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    LSTM,
    SpatialDropout1D
)

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

        model = Sequential()

        model.add(
            Embedding(
                input_dim=MAX_WORDS,
                output_dim=32,
                input_length=MAX_LEN
            )
        )

        model.add(
            SpatialDropout1D(0.2)
        )

        model.add(
            LSTM(
                32,
                dropout=0.2,
                recurrent_dropout=0.2
            )
        )

        model.add(
            Dense(
                1,
                activation=ACTIVATION
            )
        )

        model.compile(
            loss=LOSS,
            optimizer=RMSprop(),
            metrics=METRICS
        )

        model.summary()

        return model