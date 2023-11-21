import os
from argparse import ArgumentParser

import numpy as np

import experiments.base as ex
import timexplain as tx


def run_script():
    parser = ArgumentParser(description="Compute impacts en masse using various explanation methods.")
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
                        choices=ex.EXPLAINER_NAMES, default=ex.EXPLAINER_NAMES,
                        help="If supplied, limit the explainers that will generate impacts. "
                             "Available explainers are: " + ", ".join(ex.EXPLAINER_NAMES))
    parser.add_argument("-l", metavar="link", dest="link_name", choices=["identity", "logit"], default="identity",
                        help="For SHAP-based explainers, this link model ('identity' or 'logit') connects "
                             "the impacts to the model output. Default is 'identity'. Although 'logit' would "
                             "in general be a better choice than 'identity' for a classifier model, it is "
                             "unstable when the model output is very close to 0 or 1 and is even undefined "
                             "for model output equal to 0 or 1. As such, only use 'logit' when the model "
                             "output is sigmoid-like.")
    parser.add_argument("-s", metavar="#specimens", dest="n_specimens", type=int, default=5,
                        help="Compute impacts for this number of specimens per dataset, chosen "
                             "randomly and evenly distributed over all classes. Default is 5.")
    parser.add_argument("-sr", metavar="#specimen_reps", dest="n_specimen_reps", type=int, default=1,
                        help="For each specimen, repeat the computation of impacts this many times, "
                             "saving all individual results. Default is 1.")

    args = parser.parse_args()

    jobs = []
    for dataset_name in ex.script_utils.resolve_dataset_names(args.archive_name, args.dataset_names):
        for clf_name in args.clf_names:
            for clf_rep in ex.prefitted_tab(args.archive_name, dataset_name, clf_name):
                if args.max_clf_reps is None or int(clf_rep) < args.max_clf_reps:
                    for explainer_name in args.explainer_names:
                        for specimen_iter in range(args.n_specimens):
                            for specimen_rep in range(args.n_specimen_reps):
                                jobs.append({
                                    "archive_name": args.archive_name,
                                    "dataset_name": dataset_name,
                                    "clf_name": clf_name,
                                    "clf_rep": clf_rep,
                                    "explainer_name": explainer_name,
                                    "link_name": args.link_name,
                                    "specimen_iter": specimen_iter,
                                    "specimen_rep": specimen_rep
                                })

    ex.script_utils.parallel(args.max_workers, jobs, worker, "impact_logs", job_to_logfile, job_to_msg)


def job_to_logfile(archive_name, dataset_name, clf_name, clf_rep,
                   explainer_name, link_name, specimen_iter, specimen_rep):
    return f"{archive_name}-{dataset_name}-{clf_name}#{clf_rep}-{link_name}link-" \
           f"{explainer_name}-iter{specimen_iter}#{specimen_rep}.log"


def job_to_msg(archive_name, dataset_name, clf_name, clf_rep,
               explainer_name, link_name, specimen_iter, specimen_rep):
    return f"Computing impacts with explainer {explainer_name} using classifier {clf_name}#{clf_rep} " \
           f"({link_name} link) on dataset {archive_name} {dataset_name}, iteration {specimen_iter}, " \
           f"for the #{specimen_rep} time"


def worker(archive_name, dataset_name, clf_name, clf_rep,
           explainer_name, link_name, specimen_iter, specimen_rep):
    print(job_to_msg(archive_name, dataset_name, clf_name, clf_rep,
                     explainer_name, link_name, specimen_iter, specimen_rep))

    print("Will now begin loading the dataset...")
    y_train, X_test, y_test = ex.dataset(archive_name, dataset_name, X_train=False)
    y_labels = np.unique(y_train)
    print("Successfully loaded the dataset")

    specimen_idx = ex.script_utils.find_specimen_idx(y_train, y_test, specimen_iter)
    x_specimen = X_test[specimen_idx]
    y_specimen = y_test[specimen_idx]
    print(f"This iteration will examine a specimen with label {y_specimen}")
    print(f"Selected time series #{specimen_idx} from test set as specimen based on reproducible RNG")

    impacts_file = ex.impacts_path(archive_name, dataset_name, clf_name, clf_rep,
                                   explainer_name, specimen_idx, specimen_rep)
    impacts_per_bgclass_file = ex.impacts_path(archive_name, dataset_name, clf_name, clf_rep,
                                               explainer_name, specimen_idx, specimen_rep,
                                               per_bgclass=True)
    if os.path.isfile(impacts_file) and \
            (not explainer_name.endswith("_bgcs") or os.path.isfile(impacts_per_bgclass_file)):
        print("These impacts have already been computed in the past. Skipping.")
        raise ex.parallel.Skip

    print("Will now begin loading the classifier...")
    model = ex.prefitted(archive_name, dataset_name, clf_name, clf_rep)
    print("Successfully loaded the classifier")

    print("Will now begin computing the impacts...")

    if explainer_name.startswith("shap_"):
        impacts, impacts_per_bgclass = \
            shap_impacts(explainer_name, link_name, model, x_specimen, X_test, y_test, y_labels)
    else:
        impacts = spec_impacts(explainer_name, model, x_specimen, X_test)
        impacts_per_bgclass = None

    # Store impacts
    os.makedirs(os.path.dirname(impacts_file), exist_ok=True)
    np.savetxt(impacts_file + ".part", impacts,
               fmt="%.8f", delimiter=",", header="labels: " + ",".join(map(str, y_labels)))
    os.rename(impacts_file + ".part", impacts_file)
    if impacts_per_bgclass is not None:
        np.save(impacts_per_bgclass_file + ".part.npy", impacts_per_bgclass)
        os.rename(impacts_per_bgclass_file + ".part.npy", impacts_per_bgclass_file)

    print("Successfully computed and stored the impacts")
    print("Done!")


def shap_impacts(explainer_name, link_name, model, x_specimen, X_test, y_test, y_labels):
    # Construct omitter
    omitter = None
    size_x = len(x_specimen)
    if explainer_name.startswith("shap_timeslice_"):
        if explainer_name.startswith("shap_timeslice_local_mean"):
            x_repl = tx.om.x_local_mean
        elif explainer_name.startswith("shap_timeslice_global_mean"):
            x_repl = tx.om.x_global_mean
        elif explainer_name.startswith("shap_timeslice_local_noise"):
            x_repl = tx.om.x_local_noise
        elif explainer_name.startswith("shap_timeslice_global_noise"):
            x_repl = tx.om.x_global_noise
        elif explainer_name.startswith("shap_timeslice_sample"):
            x_repl = tx.om.x_sample
        omitter = tx.om.TimeSliceOmitter(size_x, tx.Slicing(n_slices=min(30, size_x // 5)), x_repl)
    elif explainer_name.startswith("shap_freqslice_"):
        t_slicing = tx.Slicing(n_slices=min(10, size_x // 15))
        if explainer_name == "shap_freqslice_firls":
            f_slicing = tx.Slicing(n_slices=min(5, size_x // 30, tx.om.FreqDiceFilterOmitter.max_n_freq_slices(size_x)))
            omitter = tx.om.FreqDiceFilterOmitter(size_x, t_slicing, f_slicing, tx.om.firls_filter)
        else:
            if explainer_name.startswith("shap_freqslice_local_mean"):
                x_patch = tx.om.x_local_mean
            elif explainer_name.startswith("shap_freqslice_sample"):
                x_patch = tx.om.x_sample
            f_slicing = tx.Slicing(n_slices=min(5, size_x // 30, tx.om.FreqDicePatchOmitter.max_n_freq_slices(size_x)))
            omitter = tx.om.FreqDicePatchOmitter(size_x, t_slicing, f_slicing, x_patch)
    elif explainer_name.startswith("shap_statistic_"):
        if explainer_name.startswith("shap_statistic_global"):
            stats_repl = tx.om.stats_global
        elif explainer_name.startswith("shap_statistic_sample"):
            stats_repl = tx.om.stats_sample
        omitter = tx.om.StatisticOmitter(stats_repl)

    def get_impacts(explanation):
        return explanation.slice_impacts if hasattr(explanation, "slice_impacts") else explanation.impacts

    # Compute impacts
    if explainer_name.endswith("_bgcs"):
        expl, expl_per_bgclass = tx.om.KernelShapExplainer(omitter, model.predict_proba, X_test, y_test, bgcs=True,
                                                           max_n_bgclasses=10, n_builds=10,
                                                           n_samples=1000 // len(y_labels), link=link_name,
                                                           pbar=True).explain_per_bgclass(x_specimen)
        impacts = get_impacts(expl)
        # Convert the dict to an array
        impacts_per_bgclass = np.array([get_impacts(expl_per_bgclass[y]) for y in y_labels])
    else:
        expl = tx.om.KernelShapExplainer(omitter, model.predict_proba, X_test, bgcs=False, n_builds=10,
                                         n_samples=1000, link=link_name, pbar=True).explain(x_specimen)
        impacts = get_impacts(expl)
        impacts_per_bgclass = None

    return impacts, impacts_per_bgclass


def spec_impacts(explainer_name, model, x_specimen, X_test):
    X_specimens = x_specimen[np.newaxis]
    if explainer_name == "tree_shap":
        if type(model).__name__ != "RotationForestClassifier":
            raise ex.parallel.Skip
        return tx.spec.TreeShapExplainer(model, X_bg=X_test).explain(X_specimens)[0].impacts
    elif explainer_name == "neural_cam":
        if not isinstance(model, ex.classifiers.KerasWrapper):
            raise ex.parallel.Skip
        return tx.spec.NeuralCamExplainer(model.estimator).explain(model.prepare_X(X_specimens))[0].impacts
    elif explainer_name.startswith("shapelet_superpos"):
        if not isinstance(model, ex.classifiers.ShapeletTransformClassifier):
            raise ex.parallel.Skip
        explainer = tx.SuperposExplainer(
            tx.spec.ShapeletTransformExplainer(model["st"]),
            tx.spec.TreeShapExplainer(model["rf"]),
            divide_spread=explainer_name.endswith("_divide_spread"))
        return explainer.explain(model["transform"].transform(X_specimens))[0].impacts
    elif explainer_name.startswith("sax_vsm_word_superpos"):
        if not isinstance(model, ex.classifiers.SAXVSMEnsembleClassifier):
            raise ex.parallel.Skip
        divide_spread = explainer_name.endswith("_divide_spread")
        explainer = tx.MeanExplainer([tx.spec.SaxVsmWordSuperposExplainer(sub["sax_vsm"],
                                                                          divide_spread=divide_spread)
                                      for sub in model.estimators_])
        return explainer.explain(X_specimens)[0].impacts
    elif explainer_name.startswith("weasel_word_superpos"):
        if not isinstance(model, ex.classifiers.WEASELEnsembleClassifier):
            raise ex.parallel.Skip
        explainer = tx.MeanExplainer([
            tx.SuperposExplainer(
                tx.spec.WeaselExplainer(sub["weasel"]),
                tx.spec.LinearShapExplainer(sub["logreg"], X_bg=sub["weasel"].transform(X_test)),
                divide_spread=explainer_name.endswith("_divide_spread"))
            for sub in model.estimators_])
        impacts = explainer.explain(X_specimens)[0].impacts
        # Fix that linear SHAP sometimes returns impacts only for one model output.
        if impacts.ndim == 1:
            impacts = np.array([-impacts, impacts])
        return impacts
