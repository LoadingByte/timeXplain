import os
from argparse import ArgumentParser

import joblib
import numpy as np

import experiments.base as ex
import timexplain as tx


def run_script():
    parser = ArgumentParser(description="Compute impact fidelity samples or informativeness curves en masse.")
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
    parser.add_argument("-cr", metavar="#clf_repetitions", dest="max_clf_reps", type=int, default=None,
                        help="Repeat the computation for these many distinct classifier instances, "
                             "trained on the same dataset. By default, all available instances are used.")
    parser.add_argument("-e", metavar="explainer", dest="explainer_names", nargs="*",
                        choices=ex.TIME_EXPLAINER_NAMES, default=ex.TIME_EXPLAINER_NAMES,
                        help="If supplied, limit the explainers whose impacts will be analyzed. "
                             "Available explainers are: " + ", ".join(ex.TIME_EXPLAINER_NAMES))
    parser.add_argument("-s", metavar="#specimens", dest="max_specimens", type=int, default=None,
                        help="Analyze this number of specimens per dataset. "
                             "By default, all specimen for which impacts are available are used.")
    parser.add_argument("-sr", metavar="#specimen_reps", dest="max_specimen_reps", type=int, default=None,
                        help="Analyze this number of repetitions per specimen. "
                             "By default, all specimen repetitions for which impacts are available are used.")
    parser.add_argument("-v", metavar="evaluator", dest="evaluator_names", nargs="*",
                        choices=ex.EVALUATOR_NAMES, default=ex.EVALUATOR_NAMES,
                        help="If supplied, limit the used evaluators to this list. "
                             "Available evaluators are: " + ", ".join(ex.EVALUATOR_NAMES))

    args = parser.parse_args()

    jobs = []
    for dataset_name in ex.script_utils.resolve_dataset_names(args.archive_name, args.dataset_names):
        for clf_name in args.clf_names:
            for clf_rep in ex.impacts_tab(args.archive_name, dataset_name, clf_name):
                if args.max_clf_reps is None or int(clf_rep) < args.max_clf_reps:
                    for explainer_name in args.explainer_names:
                        for evaluator_name in args.evaluator_names:
                            jobs.append({
                                "archive_name": args.archive_name,
                                "dataset_name": dataset_name,
                                "clf_name": clf_name,
                                "clf_rep": clf_rep,
                                "explainer_name": explainer_name,
                                "max_specimens": args.max_specimens,
                                "max_specimen_reps": args.max_specimen_reps,
                                "evaluator_name": evaluator_name
                            })

    ex.script_utils.parallel(args.max_workers, jobs, worker, "evaluation_data_logs", job_to_logfile, job_to_msg)


def job_to_logfile(archive_name, dataset_name, clf_name, clf_rep, explainer_name,
                   max_specimens, max_specimen_reps, evaluator_name):
    return f"{archive_name}-{dataset_name}-{clf_name}#{clf_rep}-{explainer_name}-{evaluator_name}.log"


def job_to_msg(archive_name, dataset_name, clf_name, clf_rep, explainer_name,
               max_specimens, max_specimen_reps, evaluator_name):
    return f"Computing {evaluator_name} data with explainer {explainer_name} using " \
           f"classifier {clf_name}#{clf_rep} on dataset {archive_name} {dataset_name}"


def worker(archive_name, dataset_name, clf_name, clf_rep, explainer_name,
           max_specimens, max_specimen_reps, evaluator_name):
    print(job_to_msg(archive_name, dataset_name, clf_name, clf_rep, explainer_name,
                     max_specimens, max_specimen_reps, evaluator_name))

    print("Will now begin loading the dataset...")
    y_train, X_test, y_test = ex.dataset(archive_name, dataset_name, X_train=False)
    print("Successfully loaded the dataset")

    print("Will now begin loading the classifier...")
    model = ex.prefitted(archive_name, dataset_name, clf_name, clf_rep)
    print("Successfully loaded the classifier")

    computed_anything = False

    avail_spec_indices = [int(s) for s in ex.impacts_tab(archive_name, dataset_name, clf_name, clf_rep, explainer_name)]
    if max_specimens is None:
        specimen_indices = avail_spec_indices
    else:
        specimen_indices = {ex.script_utils.find_specimen_idx(y_train, y_test, itr) for itr in range(max_specimens)} \
            .intersection(avail_spec_indices)

    for specimen_idx in specimen_indices:
        x_specimen = X_test[specimen_idx]

        specimen_reps = \
            [int(rep)
             for rep in ex.impacts_tab(archive_name, dataset_name, clf_name, clf_rep, explainer_name, specimen_idx)
             if int(rep) < (max_specimen_reps or np.inf)]

        for specimen_rep in specimen_reps:
            print(f"Will now commence with specimen {specimen_idx} repetition #{specimen_rep}")

            output_file = ex.evaluation_data_path(archive_name, dataset_name, clf_name, clf_rep, explainer_name,
                                                  specimen_idx, specimen_rep, evaluator_name)
            if os.path.isfile(output_file):
                print("This evaluation data has already been computed in the past. Skipping.")
                continue
            else:
                computed_anything = True

            print("Will now begin loading the impacts...")
            impacts = ex.impacts(archive_name, dataset_name, clf_name, clf_rep,
                                 explainer_name, specimen_idx, specimen_rep)
            size_x = len(x_specimen)
            if "shap" in explainer_name:
                slicing = tx.Slicing(bin_rate=1, n_slices=impacts.shape[1], bin_interval=(0, size_x))
                explanation = tx.TimeExplanation(x_specimen, impacts, time_slicing=slicing)
            else:
                explanation = tx.TimeExplanation(x_specimen, impacts)
            print("Successfully loaded the impacts")

            print("Will now begin computing the evaluation data...")

            if evaluator_name == "fidelity":
                auc, curves, points = tx.dtw_interval_fidelity(model.predict_proba, explanation, X_test,
                                                               interval_sizes=10, n_intervals_per_interval_size=50,
                                                               n_countersamples_per_interval=1, warp_window_size=0.1,
                                                               return_curves=True, return_points=True)
                output = {"auc": auc, "curves": curves, "points": points}
            else:
                auc, curves = tx.single_specimen_informativeness_eloss(model.predict_proba, explanation, X_test,
                                                                       n_perturbations=100, return_curves=True)
                output = {"auc": auc, "curves": curves}

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            joblib.dump(output, output_file + ".part")
            os.rename(output_file + ".part", output_file)

            print("Successfully computed and stored the evaluation data")

    print("Done!")

    if not computed_anything:
        raise ex.parallel.Skip
