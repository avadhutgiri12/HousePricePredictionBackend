import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=float)

        rooms_ix = 3
        bedrooms_ix = 4
        population_ix = 5
        households_ix = 6

        rooms_per_household = (
            X[:, rooms_ix] /
            np.maximum(X[:, households_ix], 1e-12)
        )

        population_per_household = (
            X[:, population_ix] /
            np.maximum(X[:, households_ix], 1e-12)
        )

        if self.add_bedrooms_per_room:
            bedrooms_per_room = (
                X[:, bedrooms_ix] /
                np.maximum(X[:, rooms_ix], 1e-12)
            )

            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room,
            ]

        return np.c_[
            X,
            rooms_per_household,
            population_per_household,
        ]