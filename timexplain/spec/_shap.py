from typing import Any, Dict

import numpy as np
import shap

from timexplain._explainer import Explainer
from timexplain._explanation import TabularExplanation


class LinearShapExplainer(Explainer[TabularExplanation]):
    model: Any
    X_bg: Any
    constr_kwargs: Dict[str, Any]

    def __init__(self, model, X_bg=None, **constr_kwargs):
        self.model = model
        self.X_bg = X_bg
        self.constr_kwargs = constr_kwargs

    def _explain(self, X_specimens):
        X_bg = _get_X_bg(self.X_bg, X_specimens)
        impacts = np.asarray(shap.LinearExplainer(self.model, X_bg, **self.constr_kwargs)
                             .shap_values(X_specimens))
        return [TabularExplanation(X_specimens[idx], impacts[..., idx, :]) for idx in range(X_specimens.shape[0])]


class TreeShapExplainer(Explainer[TabularExplanation]):
    model: Any
    X_bg: Any
    constr_kwargs: Dict[str, Any]
    shapva_kwargs: Dict[str, Any]

    def __init__(self, model, X_bg=None, **kwargs):
        self.model = model
        self.X_bg = X_bg

        shapva_keys = ["tree_limit", "approximate", "check_additivity"]
        self.constr_kwargs = {k: v for k, v in kwargs.items() if k not in shapva_keys}
        self.shapva_kwargs = {k: v for k, v in kwargs.items() if k in shapva_keys}

    def _explain(self, X_specimens):
        model = self.model
        model_name = type(model).__name__
        if model_name == "RotationForest":
            # Because SHAP values are additive, the SHAP values of the tree ensemble are equal to the mean of
            # the SHAP values of each tree.
            impacts = np.mean([self._explain_sktime_rot_tree(tree, groups, pcas, X_specimens)
                               for tree, groups, pcas in zip(model.estimators_, model._groups, model._pcas)], axis=0)
        # Support for: https://github.com/joshloyal/RotationForest
        elif model_name == "RotationTreeClassifier":
            impacts = self._explain_josh_rot_tree(model, X_specimens)
        elif model_name == "RotationForestClassifier":
            # As above, we exploit that SHAP values are additive.
            impacts = np.mean([self._explain_josh_rot_tree(rot_tree, X_specimens)
                               for rot_tree in model.estimators_], axis=0)
        else:
            impacts = self._explain_tree(model, X_specimens, _get_X_bg(self.X_bg, X_specimens))
        return [TabularExplanation(X_specimens[idx], impacts[..., idx, :]) for idx in range(X_specimens.shape[0])]

    def _explain_sktime_rot_tree(self, rot_tree, groups, pcas, X_specimens):
        rot_X_specimens = TreeShapExplainer._rotate_features_sktime(pcas, groups, X_specimens)
        rot_X_bg = TreeShapExplainer._rotate_features_sktime(pcas, groups, _get_X_bg(self.X_bg, X_specimens))
        rot_impacts = self._explain_tree(rot_tree, rot_X_specimens, rot_X_bg)

        impacts = np.zeros((*rot_impacts.shape[:-1], X_specimens.shape[-1]))
        split_indices = np.cumsum([pca.components_.shape[0] for pca in pcas])
        for pca, group, rot_group_impacts in zip(pcas, groups, np.split(rot_impacts, split_indices, axis=-1)):
            # We can just add because SHAP values are additive.
            impacts[..., group] += pca.inverse_transform(rot_group_impacts)
        return impacts

    # Taken from RotationForest._predict_proba_for_estimator()
    @staticmethod
    def _rotate_features_sktime(pcas, groups, X):
        rot_X = np.concatenate([pca.transform(X[:, group]) for pca, group in zip(pcas, groups)], axis=1)
        return np.nan_to_num(rot_X, False, 0, 0, 0)

    def _explain_josh_rot_tree(self, rot_tree, X_specimens):
        X_bg = _get_X_bg(self.X_bg, X_specimens)
        rot_matrix = rot_tree.rotation_matrix
        rot_impacts = self._explain_tree(rot_tree, X_specimens @ rot_matrix, X_bg @ rot_matrix)
        return rot_impacts @ rot_matrix.T

    def _explain_tree(self, tree, X_specimens, X_bg):
        return np.asarray(shap.TreeExplainer(tree, X_bg, **self.constr_kwargs)
                          .shap_values(X_specimens, **self.shapva_kwargs))


def _get_X_bg(X_bg, X_specimens):
    if X_bg is None and X_specimens.shape[0] >= 20:
        return X_specimens
    else:
        return X_bg
