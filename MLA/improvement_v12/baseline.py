"""Exact frozen V7 Trial 1 fitting, independent of previous project directories."""
import numpy as np
import legacy_reference as ref
import legacy_models as legacy

PORTAL_RMSE = 12024316.5355221
CHOICE = {'name': 'residual_tree_l2', 'family': 'residual', 'weight': .75}


def fold_prediction(train, y, valid, contract, fold, progress=None):
    b = legacy.v5_fold_prediction(train, y, valid, contract, fold)
    if progress:
        progress('v5 outer fit complete; generating nested residual inputs')
    inner = legacy.inner_v5_oof(train, y, contract, 731 + fold, progress)
    models = [legacy.fit_component(train, y, legacy.SPECS[CHOICE['name']], seed + fold, inner)
              for seed in legacy.SEEDS]
    p = .25 * b + .75 * np.mean([legacy.predict_component(m, valid, b) for m in models], axis=0)
    return p


def fit_full(train, test, y, contract, progress=None):
    inner = legacy.inner_v5_oof(train, y, contract, 731, progress)
    p, b, fitted = legacy.fit_submission(train, test, y, contract, CHOICE, inner)
    return p, fitted


def predict(fitted, raw):
    return legacy.predict_submission(fitted, raw)
