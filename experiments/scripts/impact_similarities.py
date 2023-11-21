from itertools import combinations, product

import numpy as np
import pandas as pd
from tqdm import tqdm

import experiments.base as ex
import timexplain as tx


def run_script():
    n_jobs = sum(len(ex.impacts_tab(archive_name))
                 for archive_name in ex.impacts_tab())

    aggr = []
    with tqdm(total=n_jobs, unit="dataset") as pbar:
        loop(pbar, aggr)

    df = pd.DataFrame(aggr, columns=["archive", "dataset", "classifier_1", "classifier_1_repetition", "classifier_2",
                                     "classifier_2_repetition", "explainer", "specimen_idx",
                                     "classifier_1_specimen_repetition", "classifier_2_specimen_repetition",
                                     "correlation_similarity"])
    df.to_csv(ex.impact_similarities_path(), float_format="%.8f", index=False)


def loop(pbar, aggr):
    for archive_name in ex.impacts_tab():
        for dataset_name in ex.impacts_tab(archive_name):
            y_train, y_test = ex.dataset(archive_name, dataset_name, X_train=False, X_test=False)
            unique_labels = np.unique(y_train)
            classifier_names = ex.impacts_tab(archive_name, dataset_name)
            for clf_1, clf_2 in combinations(classifier_names, 2):
                clf_1_reps = ex.impacts_tab(archive_name, dataset_name, clf_1)
                clf_2_reps = ex.impacts_tab(archive_name, dataset_name, clf_2)
                for clf_1_rep, clf_2_rep in product(clf_1_reps, clf_2_reps):
                    for explainer_name in shared_impacts_tab(archive_name, dataset_name, clf_1, clf_2, clf_1_rep,
                                                             clf_2_rep):
                        for specimen_idx in shared_impacts_tab(archive_name, dataset_name, clf_1, clf_2, clf_1_rep,
                                                               clf_2_rep, explainer_name):
                            c1_spec_reps = ex.impacts_tab(archive_name, dataset_name, clf_1, clf_1_rep, explainer_name,
                                                          specimen_idx)
                            c2_spec_reps = ex.impacts_tab(archive_name, dataset_name, clf_2, clf_2_rep, explainer_name,
                                                          specimen_idx)
                            for clf_1_spec_rep, clf_2_spec_rep in product(c1_spec_reps, c2_spec_reps):
                                similarity = compute_similarity(unique_labels, y_test,
                                                                archive_name, dataset_name, clf_1, clf_2, clf_1_rep,
                                                                clf_2_rep, explainer_name, specimen_idx,
                                                                clf_1_spec_rep, clf_2_spec_rep)
                                aggr.append([archive_name, dataset_name, clf_1, clf_1_rep, clf_2, clf_2_rep,
                                             explainer_name, specimen_idx, clf_1_spec_rep, clf_2_spec_rep, similarity])

            pbar.update()


def shared_impacts_tab(archive_name, dataset_name, clf_1, clf_2, clf_1_rep, clf_2_rep,
                       explainer_name=None, specimen_idx=None):
    next_1 = ex.impacts_tab(archive_name, dataset_name, clf_1, clf_1_rep, explainer_name, specimen_idx)
    next_2 = ex.impacts_tab(archive_name, dataset_name, clf_2, clf_2_rep, explainer_name, specimen_idx)
    return set(next_1) & set(next_2)


def compute_similarity(unique_labels, y_test,
                       archive_name, dataset_name, clf_1, clf_2, clf_1_rep, clf_2_rep,
                       explainer_name, specimen_idx, clf_1_spec_rep, clf_2_spec_rep):
    specimen_label_idx = np.argwhere(unique_labels == y_test[int(specimen_idx)])[0, 0]

    impacts_1 = ex.impacts(archive_name, dataset_name, clf_1, clf_1_rep,
                           explainer_name, specimen_idx, clf_1_spec_rep)[specimen_label_idx]
    impacts_2 = ex.impacts(archive_name, dataset_name, clf_2, clf_2_rep,
                           explainer_name, specimen_idx, clf_2_spec_rep)[specimen_label_idx]

    return tx.correlation(impacts_1, impacts_2)
