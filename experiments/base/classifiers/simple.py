from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sktime.classification.distance_based import ElasticEnsemble, ProximityForest
from sktime.classification.interval_based import TimeSeriesForestClassifier as _TsF, RandomIntervalSpectralEnsemble
from sktime.classification.sklearn import RotationForest
from sktime.transformations.panel.shapelet_transform import ShapeletTransform
from tslearn.svm import TimeSeriesSVC as _TsSVC

import timexplain as tx
from experiments.base.data_formats import encode_sktime_X, encode_tslearn_X

RotationForestClassifier = RotationForest


class _BaseWrapper(Pipeline):

    def __init__(self, estimator, transform_func):
        steps = [
            ("transform", FunctionTransformer(transform_func, check_inverse=False)),
            ("estimator", estimator)
        ]
        super().__init__(steps)


class ElasticEnsembleClassifier(_BaseWrapper):

    def __init__(self, **kwargs):
        super().__init__(ElasticEnsemble(**kwargs), encode_sktime_X)


class ProximityForestClassifier(_BaseWrapper):

    def __init__(self, **kwargs):
        super().__init__(ProximityForest(**kwargs), encode_sktime_X)


class TimeSeriesForestClassifier(_BaseWrapper):

    def __init__(self, **kwargs):
        super().__init__(_TsF(**kwargs), encode_sktime_X)


class RISEClassifier(_BaseWrapper):

    def __init__(self, **kwargs):
        super().__init__(RandomIntervalSpectralEnsemble(**kwargs), encode_sktime_X)


class ShapeletTransformClassifier(Pipeline):

    def __init__(self, **kwargs):
        steps = [
            ("transform", FunctionTransformer(encode_sktime_X, check_inverse=False)),
            ("st", _ContractedShapeletTransform(**kwargs)),
            ("rf", RandomForestClassifier(n_estimators=100))
        ]
        super().__init__(steps)

    # Post-hoc replacement of ShapeletTransform.transform() by our own, substantially faster implementation.
    def predict(self, X, **predict_params):
        return self["rf"].predict(self._transform_until_rf(X), **predict_params)

    def predict_proba(self, X):
        return self["rf"].predict_proba(self._transform_until_rf(X))

    def _transform_until_rf(self, X):
        return tx.spec.align_shapelets(self["st"], self["transform"].transform(X))[0]


class _ContractedShapeletTransform(ShapeletTransform):

    def __init__(self, verbose=0, num_candidates_to_sample_per_case=20, time_contract_in_mins=60):
        self.num_candidates_to_sample_per_case = num_candidates_to_sample_per_case
        self.time_contract_in_mins = time_contract_in_mins
        super().__init__(verbose=verbose)


class TimeSeriesSVC(_BaseWrapper):

    def __init__(self, **kwargs):
        self.estimator = _TsSVC(**kwargs)
        super().__init__(self.estimator, encode_tslearn_X)
