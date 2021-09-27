"""

This script contains some simple experiments.
Using simple distrbutions to explore the different metrics.

"""

import measures as msrs
import numpy as np
import pandas as pd
import os

# Initial set-up
start_directory = os.getcwd()
date_code = "d0m0"

## ADMIN: NAME ATTEMPT !!!! ##
os.chdir(start_directory)
attempt_name = "simple_experiment_"+ date_code
final_directory = os.path.join(start_directory, attempt_name)
if not os.path.exists(final_directory):
   os.mkdir(final_directory)
os.chdir(final_directory)


def exp_simple_distr(runs = 1):

    results = np.zeros((runs*4, 4))
    results[:, 0] = results[:, 0].astype('U256')

    for i in range(runs):
        n1sd2 = np.zeros((100, 3))
        n1sd2bis = np.zeros((100, 3))
        n1sd4 = np.zeros((100, 3))
        n4sd2 = np.zeros((100, 3))
        n_2sd4 = np.zeros((100, 3))

        n1sd2[:, 0] = np.random.normal(1, 1, 100)
        n1sd2bis[:, 0] = np.random.normal(1, 1, 100)
        n1sd4[:, 0] = np.random.normal(1, 1, 100)
        n4sd2[:, 0] = np.random.normal(1, 1, 100)
        n_2sd4[:, 0] = np.random.normal(1, 1, 100)

        n1sd2[:, 1] = np.random.normal(-1, 1, 100)
        n1sd2bis[:, 1] = np.random.normal(-1, 1, 100)
        n1sd4[:, 1] = np.random.normal(-1, 1, 100)
        n4sd2[:, 1] = np.random.normal(-1, 1, 100)
        n_2sd4[:, 1] = np.random.normal(-1, 1, 100)

        n1sd2[:, 2] = np.random.normal(1, 2, 100)
        n1sd2bis[:, 2] = np.random.normal(1, 2, 100)
        n1sd4[:, 2] = np.random.normal(1, 4, 100)
        n4sd2[:, 2] = np.random.normal(4, 2, 100)
        n_2sd4[:, 2] = np.random.normal(-2, 4, 100)


        wass1, kmmd1, frech1 = msrs.get_all_scores(n1sd2, n1sd2bis)
        wass2, kmmd2, frech2 = msrs.get_all_scores(n1sd2, n1sd4)
        wass3, kmmd3, frech3 = msrs.get_all_scores(n1sd2, n4sd2)
        wass4, kmmd4, frech4 = msrs.get_all_scores(n1sd2, n_2sd4)

        wass = [wass1, wass2, wass3, wass4]
        kmmd = [kmmd1, kmmd2, kmmd3, kmmd4]
        frech = [frech1, frech2, frech3, frech4]
        dist = [991, 992, 993, 994]

        results[4*i:(4*i+4), 0] = dist
        results[4*i:(4*i+4), 1] = wass
        results[4*i:(4*i+4), 2] = kmmd
        results[4*i:(4*i+4), 3] = frech

    df_results = pd.DataFrame(results, columns = ["Distance", "Wasserstein", "kMMD", "Frechet"])
    df_results = df_results.replace({991:"N(1,2) to N(1,2)", 992:"N(1,2) to N(1,4)", 993:"N(1,2) to N(4,2)", 994:"N(1,2) to N(-2,4)"})


    return df_results


df2 = exp_simple_distr(runs = 100)
df2

df2.groupby('Distance').mean()
df2.groupby('Distance').quantile([0.025, 0.975])











