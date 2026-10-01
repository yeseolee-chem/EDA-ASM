#!/usr/bin/env python3
"""r6_nn.py — Espley's NN exactly as their code defines it (rev 6 §3-2: the same NN in all four arms).

Source: the-grayson-group/distortion-interaction_ML
  set_seeds        ml_analysis.py :66-69 / hyp_tuning.py :62-64 (numpy seed, tf.random.set_seed,
                   tf.keras.utils.set_random_seed) — call once at process start, as their scripts do
  build_from_hps   ml_analysis.py :523-542 (ELU dense + l2 + dropout, no dropout after the last hidden layer,
                   1-unit output named after the target, Adam(learning_rate), loss = metric = MAE)
  fit_predict      ml_analysis.py :544-570 (batch 32, 500 epochs, shuffle False, verbose 0, validation_data only for
                   the history; X and y standardised on the training rows, predictions inverse-transformed)
  tune             hyp_tuning.py :379-408 (search space hp_storage, same layer structure) and :436-456 (Hyperband,
                   objective [mean_absolute_error, val_mean_absolute_error], max_epochs 100, factor 3, seed 23;
                   search epochs 100 with validation_data); the best trial's values are returned. Their post-search
                   10-epoch refit (:459-468) only logs, so it is not repeated.
Runs in the Espley-pinned env (env_espley: tensorflow 2.16.1, keras 3.3.3, keras_tuner 1.4.5).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import layers, models  # noqa: E402
from keras.optimizers import Adam  # noqa: E402
from keras.regularizers import l2  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tensorflow import keras  # noqa: E402

HP_STORAGE = {"dropout_rate": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1], "neurons": [32, 64, 128, 256, 512],
              "reg_val": [2e-1, 1e-1, 1e-2, 1e-3], "learning_rate": [1e-3, 1e-4, 1e-5]}
TUNE_SEED = 23
FIT = dict(batch_size=32, epochs=500, verbose=0, shuffle=False)
HB = dict(objective=["mean_absolute_error", "val_mean_absolute_error"], max_epochs=100, factor=3, seed=TUNE_SEED)
SEARCH_EPOCHS = 100


def set_seeds(s):
    from numpy.random import seed
    seed(int(s))
    tf.random.set_seed(int(s))
    tf.keras.utils.set_random_seed(int(s))


def _layers(inp, model_size, get, target):
    layer_d = {"inp_lay": inp}
    for hidden_layers in range(1, model_size + 1):
        if hidden_layers == 1:
            layer_d["dense_1"] = layers.Dense(units=get("neurons_1"), name="dense_1", activation="elu",
                                              kernel_regularizer=l2(get(f"reg_val_{hidden_layers}")))(layer_d["inp_lay"])
            layer_d["drop_1"] = layers.Dropout(rate=get(f"dropout_rate_{hidden_layers}"), name="dropout_1")(layer_d["dense_1"])
        else:
            layer_d[f"dense_{hidden_layers}"] = layers.Dense(
                units=get(f"neurons_{hidden_layers}"), name=f"dense_{hidden_layers}", activation="elu",
                kernel_regularizer=l2(get(f"reg_val_{hidden_layers}")))(layer_d[f"drop_{hidden_layers - 1}"])
            if hidden_layers == model_size:
                layer_d["out_lay"] = layers.Dense(units=1, name=target)(layer_d[f"dense_{hidden_layers}"])
            else:
                layer_d[f"drop_{hidden_layers}"] = layers.Dropout(rate=get(f"dropout_rate_{hidden_layers}"),
                                                                  name=f"dropout_{hidden_layers}")(layer_d[f"dense_{hidden_layers}"])
    return layer_d


def build_from_hps(n_in, model_size, hps, target):
    """ml_analysis.py :523-542."""
    layer_d = _layers(keras.Input(shape=(n_in,), name="input"), model_size, lambda k: hps[k], target)
    model = models.Model(inputs=layer_d["inp_lay"], outputs=layer_d["out_lay"])
    model.compile(optimizer=Adam(learning_rate=hps["learning_rate"]), loss="mean_absolute_error",
                  metrics=["mean_absolute_error"])
    return model


def make_builder(n_in, model_size, target):
    """hyp_tuning.py :368-408 (their globals shape / network_size / target become arguments)."""
    def builder(hp):
        specific = {}

        def get(name):
            return specific[name]
        layer_d = {"inp_lay": keras.Input(shape=(n_in,), name="input")}
        for hidden_layers in range(1, model_size + 1):
            for key in HP_STORAGE:
                if key == "learning_rate":
                    specific[key] = hp.Choice(key, values=HP_STORAGE[key])
                else:
                    specific[f"{key}_{hidden_layers}"] = hp.Choice(f"{key}_{hidden_layers}", values=HP_STORAGE[key])
            if hidden_layers == 1:
                layer_d["dense_1"] = layers.Dense(units=specific["neurons_1"], name="dense_1", activation="elu",
                                                  kernel_regularizer=l2(specific[f"reg_val_{hidden_layers}"]))(layer_d["inp_lay"])
                layer_d["drop_1"] = layers.Dropout(rate=specific[f"dropout_rate_{hidden_layers}"], name="dropout_1")(layer_d["dense_1"])
            else:
                layer_d[f"dense_{hidden_layers}"] = layers.Dense(
                    units=specific[f"neurons_{hidden_layers}"], name=f"dense_{hidden_layers}", activation="elu",
                    kernel_regularizer=l2(specific[f"reg_val_{hidden_layers}"]))(layer_d[f"drop_{hidden_layers - 1}"])
                if hidden_layers == model_size:
                    layer_d["out_lay"] = layers.Dense(units=1, name=target)(layer_d[f"dense_{hidden_layers}"])
                else:
                    layer_d[f"drop_{hidden_layers}"] = layers.Dropout(
                        rate=specific[f"dropout_rate_{hidden_layers}"], name=f"dropout_{hidden_layers}")(layer_d[f"dense_{hidden_layers}"])
        model = models.Model(inputs=layer_d["inp_lay"], outputs=layer_d["out_lay"])
        model.compile(optimizer=Adam(learning_rate=specific["learning_rate"]), loss="mean_absolute_error",
                      metrics=["mean_absolute_error"])
        return model
    return builder


def _scale(Xtr, ytr, *others):
    """Their _standardise_data 'both_X_y': StandardScaler on X_train / y_train, applied to the other sets.
    DataFrames in, DataFrames / Series out (keras receives the same pandas objects as in their code)."""
    sx = StandardScaler()
    out = [pd.DataFrame(sx.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)]
    sy = StandardScaler()
    ys = pd.DataFrame(sy.fit_transform(pd.DataFrame(ytr)), columns=pd.DataFrame(ytr).columns, index=ytr.index)
    out.append(ys.iloc[:, 0])
    for X, y in others:
        out.append(pd.DataFrame(sx.transform(X), columns=X.columns, index=X.index))
        out.append(None if y is None else
                   pd.DataFrame(sy.transform(pd.DataFrame(y)), columns=pd.DataFrame(y).columns, index=y.index).iloc[:, 0])
    return out, sx, sy


def fit_predict(Xtr, ytr, Xval, yval, X_pred, model_size, hps, target):
    """ml_analysis.py :523-570 on DataFrame X / Series y. X_pred: {name: DataFrame}. Returns ({name: preds
    kcal/mol}, history dict). Call set_seeds(seed) at process start first, as their script does."""
    (Xs, ys, Xv, yv), sx, sy = _scale(Xtr, ytr, (Xval, yval))
    model = build_from_hps(Xs.shape[1], model_size, hps, target)
    hist = model.fit(x=Xs, y=ys, validation_data=(Xv, yv), **FIT)
    preds = {}
    for name, X in X_pred.items():
        Xp = pd.DataFrame(sx.transform(X), columns=X.columns, index=X.index)
        p = model.predict(Xp, verbose=0)
        preds[name] = sy.inverse_transform(np.array(p).reshape(-1, 1)).ravel()
    return preds, dict(loss=[float(v) for v in hist.history["loss"]],
                       val_loss=[float(v) for v in hist.history["val_loss"]])


def tune(Xtr, ytr, Xval, yval, model_size, target, directory, project_name):
    """hyp_tuning.py :436-456. Call set_seeds(TUNE_SEED) at process start first, as their script does."""
    import keras_tuner as kt
    (Xs, ys, Xv, yv), _, _ = _scale(Xtr, ytr, (Xval, yval))
    tuner = kt.Hyperband(make_builder(Xs.shape[1], model_size, target), project_name=project_name,
                         directory=str(directory), **HB)
    tuner.search(x=Xs, y=ys, validation_data=(Xv, yv), epochs=SEARCH_EPOCHS, verbose=0)
    best = tuner.get_best_hyperparameters(num_trials=1)[0]
    return dict(best.values)
