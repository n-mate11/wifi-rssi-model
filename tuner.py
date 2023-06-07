from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import keras_tuner as kt

MAX_EPOCHS = 205
MAX_BATCH_SIZE = 128


class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Flatten())
        # Wetther to use normalization
        if hp.Boolean("normalization"):
            model.add(BatchNormalization())

        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 5)):
            model.add(
                Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=16, max_value=512, step=32),
                    activation="relu",
                )
            )
        if hp.Boolean("dropout"):
            model.add(Dropout(rate=0.25))
        model.add(Dense(units=3, activation="linear"))

        # optimize learning rate
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
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
                "batch_size", min_value=8, max_value=MAX_BATCH_SIZE, step=8
            ),
            epochs=hp.Int("epochs", min_value=5, max_value=MAX_EPOCHS, step=10),
            shuffle=hp.Boolean("shuffle"),
        )
