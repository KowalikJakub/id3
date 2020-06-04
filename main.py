import pandas as pd

from src.id3 import *

data = pd.DataFrame(
    {
        "toothed": [
            "True",
            "True",
            "True",
            "False",
            "True",
            "True",
            "True",
            "True",
            "True",
            "False",
        ],
        "hair": [
            "True",
            "True",
            "False",
            "True",
            "True",
            "True",
            "False",
            "False",
            "True",
            "False",
        ],
        "breathes": [
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "False",
            "True",
            "True",
            "True",
        ],
        "legs": [
            "True",
            "True",
            "False",
            "True",
            "True",
            "True",
            "False",
            "False",
            "True",
            "True",
        ],
        "species": [
            "Mammal",
            "Mammal",
            "Reptile",
            "Mammal",
            "Mammal",
            "Mammal",
            "Reptile",
            "Reptile",
            "Mammal",
            "Reptile",
        ],
    },
    columns=["toothed", "hair", "breathes", "legs", "species"],
)

features = data[["toothed", "breathes", "legs"]]
target = data["species"]


model = DecisionTree()
model.fit(features.values, target.values)
