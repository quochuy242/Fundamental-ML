from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Input
from keras.optimizers import Adam
from keras.metrics import F1Score, Precision, Recall, SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Set params for GridSearchCV
dropout_rate = [0.1, 0.2, 0.5]
dense_units = [(128, 64), (256, 128)]
learning_rate = [0.001, 0.01, 0.005]

param_grid = dict(
    model__dropout_rate=dropout_rate,
    model__dense_units=dense_units,
    optimizer__learning_rate=learning_rate,
)

mlp_model = [
    Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(dense_units[0], activation="relu"),
            BatchNormalization(),
            Dense(dense_units[1], activation="relu"),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(num_classes, activation="sigmoid"),
        ],
        name=f"MLP for {name}",
    )
    for name, X_train in zip(
        ["Original Data", "Transformed Data"], [origin_X_train, transformed_X_train]
    )
]

mlp_model = [
    KerasClassifier(
        model=model,
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            SparseCategoricalAccuracy(name="accuracy"),
            F1Score(name="f1", average="micro"),
            Precision(name="pre"),
            Recall(name="recall"),
        ],
        batch_size=32,
        epochs=150,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    )
    for model in mlp_model
]

mlp_gridsearch = [
    GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    for model in mlp_model
]


for model, X_train, y_train in zip(
    mlp_gridsearch,
    [origin_X_train, transformed_X_train],
    [origin_y_train, transformed_y_train],
):
    model.fit(X_train, y_train)
    print(f"Best: {model.best_score_} using {model.best_params_}")
