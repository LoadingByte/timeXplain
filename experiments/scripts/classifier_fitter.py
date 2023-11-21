import os
from argparse import ArgumentParser

import joblib
import numpy as np

import experiments.base as ex


def run_script():
    parser = ArgumentParser(description="Fit and store classifiers en masse.")
    parser.add_argument("-p", metavar="#processes", dest="max_workers", type=int, default=2,
                        help="Initially launch this number of subprocess workers.")
    parser.add_argument("-a", metavar="archive", dest="archive_name", required=True, choices=ex.ARCHIVE_NAMES,
                        help="The data archive which supplies datasets. "
                             "Available archives are: " + ", ".join(ex.ARCHIVE_NAMES))
    parser.add_argument("-d", metavar="dataset", dest="dataset_names", nargs="*",
                        help="If supplied, limit the used datasets to this list.")
    parser.add_argument("-c", metavar="clf", dest="clf_names", nargs="*",
                        choices=ex.CLASSIFIER_NAMES, default=ex.CLASSIFIER_NAMES,
                        help="If supplied, limit the used classifiers to this list. "
                             "Available classifiers are: " + ", ".join(ex.CLASSIFIER_NAMES))
    parser.add_argument("-cr", metavar="#clf_repetitions", dest="n_clf_reps", type=int, default=1,
                        help="Train this many instances of each classifier on the same dataset. Default is 1.")

    args = parser.parse_args()

    jobs = []
    for dataset_name in ex.script_utils.resolve_dataset_names(args.archive_name, args.dataset_names):
        for clf_name in args.clf_names:
            for clf_rep in range(args.n_clf_reps):
                prefitted_exists = os.path.isfile(ex.prefitted_path(
                    args.archive_name, dataset_name, clf_name, clf_rep))
                test_pred_exists = os.path.isfile(ex.test_preds_path(
                    args.archive_name, dataset_name, clf_name, clf_rep))
                if not prefitted_exists or not test_pred_exists:
                    jobs.append({
                        "archive_name": args.archive_name,
                        "dataset_name": dataset_name,
                        "clf_name": clf_name,
                        "clf_rep": clf_rep
                    })

    ex.script_utils.parallel(args.max_workers, jobs, worker, "fitting_logs", job_to_logfile, job_to_msg)


def job_to_logfile(archive_name, dataset_name, clf_name, clf_rep):
    return f"{archive_name}-{dataset_name}-{clf_name}#{clf_rep}.log"


def job_to_msg(archive_name, dataset_name, clf_name, clf_rep):
    return f"Fitting classifier {clf_name} on dataset {archive_name} {dataset_name} for the #{clf_rep} time"


def worker(archive_name, dataset_name, clf_name, clf_rep):
    print(job_to_msg(archive_name, dataset_name, clf_name, clf_rep))

    print("Will now begin loading the dataset...")

    X_train, y_train, X_test = ex.dataset(archive_name, dataset_name, y_test=False)
    time_series_length = X_train.shape[1]

    print("Successfully loaded the dataset")
    print("Will now begin fitting the classifier...")

    model_file = ex.prefitted_path(archive_name, dataset_name, clf_name, clf_rep)
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    if clf_name == "resnet":
        model = ex.classifiers.Resnet(verbose=1)
        model.fit(X_train, y_train)
        model.estimator.save(model_file + ".part")
        os.rename(model_file + ".part", model_file)
    else:
        if clf_name == "rotation_forest":
            model = ex.classifiers.RotationForestClassifier()
        elif clf_name == "svm_linear":
            model = ex.classifiers.TimeSeriesSVC(verbose=1, kernel="linear", probability=True)
        elif clf_name == "elastic_ensemble":
            model = ex.classifiers.ElasticEnsembleClassifier(verbose=1, proportion_of_param_options=0.2,
                                                             proportion_train_in_param_finding=0.2)
        elif clf_name == "proximity_forest":
            model = ex.classifiers.ProximityForestClassifier(verbosity=1)
        elif clf_name == "time_series_forest":
            model = ex.classifiers.TimeSeriesForestClassifier()
        elif clf_name == "rise":
            model = ex.classifiers.RISEClassifier()
        elif clf_name == "shapelet_transform":
            model = ex.classifiers.ShapeletTransformClassifier(verbose=1, time_contract_in_mins=15)
        elif clf_name == "sax_vsm":
            numbers = [n for n in [6, 8, 12, 16, 20] if n <= time_series_length]
            model = ex.classifiers.SAXVSMEnsembleClassifier(n_bins=numbers, window_size=numbers, strategy=["uniform"])
        elif clf_name == "weasel":
            model = ex.classifiers.WEASELEnsembleClassifier(window_sizes=np.arange(7, time_series_length),
                                                            word_size=[2, 4, 6], norm_mean=[True, False])
        else:
            raise ValueError(f"Unknown classifier: '{clf_name}'")

        # Fit the model
        model.fit(X_train, y_train)

        # Make sure that when we use the model later on, we won't be overwhelmed by its verbosity
        if clf_name == "proximity_forest":
            model.verbosity = 0
        elif (clf_name == "resnet" or clf_name == "svm_linear" or
              clf_name == "elastic_ensemble" or clf_name == "shapelet_transform"):
            model.verbose = 0

        joblib.dump(model, model_file + ".part")
        os.rename(model_file + ".part", model_file)

    print("Successfully fitted and stored the classifier")
    print("Will now begin predicting the test dataset...")

    test_prediction = model.predict_proba(X_test)
    unique_labels = np.unique(y_train)

    test_pred_path = ex.test_preds_path(archive_name, dataset_name, clf_name, clf_rep)
    os.makedirs(os.path.dirname(test_pred_path), exist_ok=True)
    np.savetxt(test_pred_path + ".part", test_prediction,
               fmt="%.8f", delimiter=",", header="labels: " + ",".join(map(str, unique_labels)))
    os.rename(test_pred_path + ".part", test_pred_path)

    print("Successfully stored the test predictions")
    print("Done!")
