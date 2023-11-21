import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, roc_auc_score
from tqdm import tqdm

import experiments.base as ex


def run_script():
    n_jobs = sum(sum(sum(len(ex.test_preds_tab(archive_name, dataset_name, classifier_name))
                         for classifier_name in ex.test_preds_tab(archive_name, dataset_name))
                     for dataset_name in ex.test_preds_tab(archive_name))
                 for archive_name in ex.test_preds_tab())

    aggr = []
    with tqdm(total=n_jobs, unit="classifier") as pbar:
        for archive_name in ex.test_preds_tab():
            for dataset_name in ex.test_preds_tab(archive_name):
                for classifier_name in ex.test_preds_tab(archive_name, dataset_name):
                    for classifier_repetition in ex.test_preds_tab(archive_name, dataset_name, classifier_name):
                        metrics = compute_metrics(archive_name, dataset_name, classifier_name, classifier_repetition)
                        aggr.append([archive_name, dataset_name, classifier_name, classifier_repetition, *metrics])
                        pbar.update()

    df = pd.DataFrame(aggr, columns=["archive", "dataset", "classifier", "classifier_repetition",
                                     "accuracy", "precision", "recall", "f1", "log_loss", "roc_auc"])
    df.to_csv(ex.test_pred_metrics_path(), float_format="%.8f", index=False)


def compute_metrics(archive_name, dataset_name, classifier_name, classifier_repetition):
    y_train, y_test = ex.dataset(archive_name, dataset_name, X_train=False, X_test=False)
    unique_labels = np.unique(y_train)
    y_test_one_hot = one_hot(y_test, unique_labels)

    y_pred_proba = ex.test_preds(archive_name, dataset_name, classifier_name, classifier_repetition)
    y_pred = [unique_labels[i] for i in np.argmax(y_pred_proba, axis=1)]

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    ll = log_loss(y_test_one_hot, y_pred_proba)
    roc_auc = roc_auc_score(y_test_one_hot, y_pred_proba, average="macro")

    return accuracy, precision, recall, f1, ll, roc_auc


def one_hot(y, labels):
    def one_hot_single(s):
        r = np.zeros(len(labels), dtype="int")
        r[np.where(labels == s)] = 1
        return r

    return np.array([one_hot_single(s) for s in y])
