import pandas as pd
from PIL import Image, ImageDraw

VALIDATION_SPLIT = 0.2

START_COORDINATES = (10, 9, 6)
END_COORDINATES = (41, 9, 6)

# Distance in meters
DISTANCE = END_COORDINATES[0] - START_COORDINATES[0]


def preprocess_experiment_data(experiment_df):
    experiment_df["startTimestamp"] = pd.to_datetime(
        experiment_df["startTimestamp"], unit="ms"
    )
    experiment_df["stopTimestamp"] = pd.to_datetime(
        experiment_df["stopTimestamp"], unit="ms"
    )

    experiment_df["scanTime"] = pd.to_datetime(experiment_df["scanTime"], unit="ms")

    # Calculate average speed in m/s
    experiment_df["averageSpeed"] = (
        DISTANCE
        / (
            experiment_df["stopTimestamp"] - experiment_df["startTimestamp"]
        ).dt.total_seconds()
    )

    # Calculate distance to scan in meters and round to nearest integer
    experiment_df["x"] = (
        (
            (
                (
                    experiment_df["scanTime"] - experiment_df["startTimestamp"]
                ).dt.total_seconds()
            )
            * experiment_df["averageSpeed"]
        )
        .round()
        .astype(int)
    )

    experiment_df["y"] = 9
    experiment_df["z"] = 6

    experiment_df.drop(
        experiment_df.filter(regex="_timestamp").columns, axis=1, inplace=True
    )
    experiment_df.drop("averageSpeed", axis=1, inplace=True)
    experiment_df.drop("startTimestamp", axis=1, inplace=True)
    experiment_df.drop("stopTimestamp", axis=1, inplace=True)
    experiment_df.drop("scanTime", axis=1, inplace=True)

    return experiment_df


def draw_predictions_on_map(test, y_pred, name):
    # use pillow to draw the predictions on map
    im = Image.open("./6th floor drawing.png")
    draw = ImageDraw.Draw(im)

    width, height = im.size
    x_step, y_step = width / 70, height / 70
    point_size = 10

    # adding 20 coordinate values to adjust the xy coordinate to fit the map

    # draw the start point
    draw.rectangle(
        (
            (START_COORDINATES[0] * x_step - point_size + 20),
            (height - (START_COORDINATES[1] * y_step) - point_size + 20),
            (START_COORDINATES[0] * x_step + point_size + 20),
            (height - (START_COORDINATES[1] * y_step) + point_size + 20),
        ),
        fill="green",
        outline="green",
    )

    # draw the end point on map starting from the bottom left corner
    draw.rectangle(
        (
            (END_COORDINATES[0] * x_step - point_size + 20),
            (height - (END_COORDINATES[1] * y_step) - point_size + 20),
            (END_COORDINATES[0] * x_step + point_size + 20),
            (height - (END_COORDINATES[1] * y_step) + point_size + 20),
        ),
        fill="blue",
        outline="blue",
    )

    test.reset_index(drop=True, inplace=True)

    # loop through the test set and draw the points on map
    for i in range(len(test)):
        # get the coordinates of the test set
        x = test["x"][i]
        y = test["y"][i]

        # calculate the coordinates on the map
        x = x_step * x * 70 + START_COORDINATES[0] * x_step + 20
        y = height - (y_step * y * 70) + 20

        # draw the point
        draw.ellipse(
            (
                (x - point_size),
                (y - point_size),
                (x + point_size),
                (y + point_size),
            ),
            fill=(92, 92, 92),
            outline=(92, 92, 92),
        )

    # draw the predictions on map
    for i in range(len(y_pred)):
        # get the coordinates of the prediction
        x = y_pred[i][0]
        y = y_pred[i][1]

        # calculate the coordinates on the map
        x = x_step * x * 70 + START_COORDINATES[0] * x_step + 20
        y = height - (y_step * y * 70) + 20

        # draw the point
        draw.ellipse(
            (
                (x - point_size),
                (y - point_size),
                (x + point_size),
                (y + point_size),
            ),
            fill="red",
            outline="red",
        )

    im.save(f"./evaluation_images/experiment/{name}.png")
    im.show()
