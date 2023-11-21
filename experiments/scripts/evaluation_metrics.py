import numpy as np
import pandas as pd
from tqdm import tqdm

import experiments.base as ex


def run_script():
    n_jobs = sum(len(ex.evaluation_data_tab(archive_name))
                 for archive_name in ex.evaluation_data_tab())

    aggr = []
    with tqdm(total=n_jobs, unit="dataset") as pbar:
        loop(pbar, aggr)

    df = pd.DataFrame(aggr, columns=["archive", "dataset", "classifier", "classifier_repetition", "explainer",
                                     "specimen_idx", "specimen_repetition", "evaluator", "integral_up_to", "value"])
    df.to_csv(ex.evaluation_metrics_path(), float_format="%.8f", index=False)


def loop(pbar, aggr):
    for archive_name in ex.evaluation_data_tab():
        for dataset_name in ex.evaluation_data_tab(archive_name):
            y_train, y_test = ex.dataset(archive_name, dataset_name, X_train=False, X_test=False)
            y_labels = np.unique(y_train)
            for clf_name in ex.evaluation_data_tab(archive_name, dataset_name):
                for clf_rep in ex.evaluation_data_tab(archive_name, dataset_name, clf_name):
                    for explainer_name in ex.evaluation_data_tab(archive_name, dataset_name, clf_name, clf_rep):
                        for specimen_idx in ex.evaluation_data_tab(archive_name, dataset_name, clf_name, clf_rep,
                                                                   explainer_name):
                            specimen_label_idx = np.argwhere(y_labels == y_test[int(specimen_idx)])[0, 0]
                            for specimen_rep in ex.evaluation_data_tab(archive_name, dataset_name, clf_name, clf_rep,
                                                                       explainer_name, specimen_idx):
                                for evaluator_name in ex.evaluation_data_tab(archive_name, dataset_name, clf_name,
                                                                             clf_rep, explainer_name, specimen_idx,
                                                                             specimen_rep):
                                    curves = ex.evaluation_data(archive_name, dataset_name, clf_name, clf_rep,
                                                                explainer_name, specimen_idx, specimen_rep,
                                                                evaluator_name)["curves"][specimen_label_idx]

                                    if evaluator_name == "fidelity":
                                        tos, dx = range(10), 0.1
                                    else:
                                        tos, dx = range(0, 101, 10), 0.01
                                    for to in tos:
                                        if to == 0:
                                            value = curves[0] * dx
                                        else:
                                            value = np.trapz(curves[:to + 1], dx=dx)
                                        aggr.append([archive_name, dataset_name, clf_name, clf_rep, explainer_name,
                                                     specimen_idx, specimen_rep, evaluator_name, to, value])

            pbar.update()
