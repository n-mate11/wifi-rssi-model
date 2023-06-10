from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import keras_tuner as kt

MIN_EPOCHS = 5
MAX_EPOCHS = 305
STEP_EPOCHS = 10

MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 128
STEP_BATCH_SIZE = 8

MIN_LAYERS = 1
MAX_LAYERS = 5

MIN_UNITS = 8
MAX_UNITS = 512
STEP_UNITS = 8

MIN_DROPOUT = 0.0
MAX_DROPOUT = 0.5
STEP_DROPOUT = 0.05
DEFAULT_DROPOUT = 0.25

LEARNING_RATES = [1e-2, 1e-3, 1e-4]


class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Flatten())
        # Whether to use normalization
        if hp.Boolean("normalization"):
            model.add(BatchNormalization())

        # Tune the number of layers.
        for i in range(hp.Int("num_layers", MIN_LAYERS, MAX_LAYERS)):
            model.add(
                Dense(
                    # Tune number of units separately.
                    units=hp.Int(
                        f"units_{i}",
                        min_value=MIN_UNITS,
                        max_value=MAX_UNITS,
                        step=STEP_UNITS,
                    ),
                    activation="relu",
                )
            )
        if hp.Boolean("dropout"):
            model.add(
                Dropout(
                    rate=hp.Float(
                        "dropout_rate",
                        min_value=MIN_DROPOUT,
                        max_value=MAX_DROPOUT,
                        default=DEFAULT_DROPOUT,
                        step=STEP_DROPOUT,
                    )
                )
            )
        model.add(Dense(units=3, activation="linear"))

        # optimize learning rate
        hp_learning_rate = hp.Choice("learning_rate", values=LEARNING_RATES)
        model.compile(
            optimizer=Adam(learning_rate=hp_learning_rate),
            loss="mean_squared_error",
            metrics=["mse", "mae"],
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
            batch_size=hp.Int(
                "batch_size",
                min_value=MIN_BATCH_SIZE,
                max_value=MAX_BATCH_SIZE,
                step=STEP_BATCH_SIZE,
            ),
            epochs=hp.Int(
                "epochs", min_value=MIN_EPOCHS, max_value=MAX_EPOCHS, step=STEP_EPOCHS
            ),
            shuffle=hp.Boolean("shuffle"),
        )
