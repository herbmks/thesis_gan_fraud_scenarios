# -*- coding: utf-8 -*-
"""

This script includes utilities necessary to process the results data as to create the plots.

"""
import numpy as np
import pandas as pd


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_ma_losses(losses, ma = 20):
    losses_ma = np.zeros((len(moving_average(losses[:,0], ma)),3))
    losses_ma[:,0] = moving_average(losses[:,0], ma)
    losses_ma[:,1] = moving_average(losses[:,1], ma)
    losses_ma[:,2] = moving_average(losses[:,2], ma)
    return losses_ma

def get_basic_confusion_vals(classifications):
    array = np.asarray(classifications)
    splits = np.unique(array[:, 4])
    thresholds = np.unique(array[:, 5])

    results = np.zeros((0, 7))

    for split in splits:
        test_split = array[array[:, 4] == split, :]

        for threshold in thresholds:
            test = test_split[test_split[:, 5] == threshold, :]

            for metric in range(3):
                # 0 - wass, 1 - kmmd, 2 - frechet
                TP = 0
                FP = 0
                TN = 0
                FN = 0

                for obs in range(len(test)):
                    if test[obs, 3] == test[obs, metric] == 1:
                        TP += 1
                    if test[obs, metric] == 1 and test[obs, 3] != test[obs, metric]:
                        FP += 1
                    if test[obs, 3] == test[obs, metric] == 0:
                        TN += 1
                    if test[obs, metric] == 0 and test[obs, 3] != test[obs, metric]:
                        FN += 1

                performance = [split, threshold, metric, TP, FP, TN, FN]

                results = np.vstack((results, performance))

    pd_results = pd.DataFrame(results, columns = ["Split", "Threshold", "Measure", "TP", "FP", "TN", "FN"])
    pd_results["Measure"].replace({0:"Wasserstein", 1:"kMMD", 2:"Frechet"}, inplace = True)

    return pd_results

def convert_confusion_to_long(pd_confusion):
    confusion2 = pd_confusion.to_numpy()
    confusion3 = np.empty((0, 5), dtype = "object")
    for acc in range(3, 7):

        new = np.zeros((len(confusion2), 5))
        new[:, 0:2] = confusion2[:, 0:2]
        new[confusion2[:, 2] == 'Wasserstein', 2] = 0
        new[confusion2[:, 2] == 'kMMD', 2] = 1
        new[confusion2[:, 2] == 'Frechet', 2] = 2
        new[:, 4] = confusion2[:, acc]

        if acc == 3:
            new[:, 3] = 0 #TP
        if acc == 4:
            new[:, 3] = 1 #FP
        if acc == 5:
            new[:, 3] = 2 #"TN"
        if acc == 6:
            new[:, 3] = 3 #"FN"

        confusion3 = np.vstack((confusion3, new))

    pd_confusion3 = pd.DataFrame(confusion3, columns = ["Split", "Threshold", "Measure", "Acc. Metric", "Result"])
    pd_confusion3["Measure"].replace({0:"Wasserstein", 1:"kMMD", 2:"Frechet"}, inplace = True)
    pd_confusion3["Acc. Metric"].replace({0:"True Pos", 1:"False Pos", 2:"True Neg", 3:"False Neg"}, inplace = True)
    pd_confusion3["Split"] = pd_confusion3["Split"].astype("category")
    pd_confusion3["Threshold"] = pd_confusion3["Threshold"].astype("category")
    pd_confusion3["Result"] = pd_confusion3["Result"].astype(float)

    return pd_confusion3

def get_adv_confusion_vals(confusion):
    test = confusion
    test["Precision"] = confusion["TP"]/(confusion["TP"] + confusion["FP"])
    test["Recall"] = confusion["TP"]/(confusion["TP"] + confusion["FN"])
    test["FP rate"] = confusion["FP"]/(confusion["FP"] + confusion["TN"])
    test["TP rate"] = confusion["TP"]/(confusion["TP"] + confusion["FN"])
    test["F score"] = 2*((test["Precision"] * test["Recall"])/(test["Precision"] + test["Recall"]))

    return test


def exp2_get_long_results(reference, observed):
    splits = np.unique(reference[:,3])
    full_results = np.zeros((0,4))
    for i in range(3):
        ref_red = reference[reference[:,3]==splits[i], :]
        obs_red = observed[observed[:, 4]==splits[i], :]
        obs_true = obs_red[obs_red[:, 3]==0, :]
        obs_false = obs_red[obs_red[:, 3]==1, :]
        
        wass_ref = ref_red[:,0]
        wass_obs_true = obs_true[:,0]
        wass_obs_false = obs_false[:,0]
    
        kmmd_ref = ref_red[:,1]
        kmmd_obs_true = obs_true[:,1]
        kmmd_obs_false = obs_false[:,1]
    
        frch_ref = ref_red[:,2]
        frch_obs_true = obs_true[:,2]
        frch_obs_false = obs_false[:,2]
    
        name_ref = np.repeat('known_true', 1000)
        name_obs_true = np.repeat('unknown_true', 50)
        name_obs_false = np.repeat('unknown_false', 50)
    
        names = np.hstack((name_ref, name_obs_true, name_obs_false))
        
        wass_dist = np.hstack((wass_ref, wass_obs_true, wass_obs_false))
        kmmd_dist = np.hstack((kmmd_ref, kmmd_obs_true, kmmd_obs_false))
        frch_dist = np.hstack((frch_ref, frch_obs_true, frch_obs_false))
        
        dist_names = np.hstack((np.repeat('Wasserstein', 1100), np.repeat('Kernel MMD', 1100), np.repeat('Frechet', 1100)))
        obs_names = np.hstack((names, names, names))
        distances = np.hstack((wass_dist, kmmd_dist, frch_dist))
        split_val = np.repeat(splits[i], len(dist_names))
        
        results = np.vstack((distances, obs_names, dist_names, split_val)).transpose()
        
        full_results = np.vstack((full_results, results))
    
    return full_results