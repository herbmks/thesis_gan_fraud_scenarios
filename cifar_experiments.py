# -*- coding: utf-8 -*-
"""
This script inludes the CIFAR experiments.
The experiments aims to investigate the behaviour of different distance measures
    in a fraud detection setting.v
For the feature representation, we use the feature space of GAN discriminators.
    These GANs and their feature spaces are accessible fron a seperate script.
> CIFAR_gans.py
For the distance measures, we use those accessible from a different script.
> measures.py
"""

import tensorflow.keras as keras
import pandas as pd
import numpy as np
import measures as msrs
from sklearn.decomposition import PCA
import umap

class MultipleObs():
    """
    Performs an experiment investigating the distance measure behaviour for different levels of fraud.
    Uses CIFAR10 as a source for true observations and CIFAR100 as a source for fraudulent observations.
    Samples of a specified size are compared: one is known, the other unknown with a specified proportion of fraud.
    """

    def __init__(self, model):

        self.known_true, self.unknown_true, self.unknown_false = self._get_data()
        self.transformer = model

    def _get_data(self):
        """Obtains the data and performs necessary preprocessing"""
        (raw_known_true,_), (raw_unknown_true,_) = keras.datasets.cifar10.load_data()
        (_,_), (raw_unknown_false,_) = keras.datasets.cifar100.load_data()

        known_true = raw_known_true.astype('float32')
        unknown_true = raw_unknown_true.astype('float32')
        unknown_false = raw_unknown_false.astype('float32')

        del raw_known_true, raw_unknown_true, raw_unknown_false

        known_true = (known_true - 127.5) - 127.5
        unknown_true = (unknown_true - 127.5) - 127.5
        unknown_false = (unknown_false - 127.5) - 127.5

        return known_true, unknown_true, unknown_false

    def _get_features(self, subset):
        """Obtains the representation of the observations in the features space of the provided transformer model"""
        # Do them all at once for faster computation.
        features = self.transformer.predict(subset)
        mx_features = np.asarray(features)

        return mx_features

    def run_experiment(self, proportions, size = 100, runs = None, seedval = None, pdtable = True):
        """
        Runs the experiment
        use: init > run_experiment
        """
        if runs == None:
            runs = abs(len(self.unknown_true)//size)
        print("Number of runs: ", runs)
        np.random.seed(seed = seedval)
        results = np.zeros((0,5))
        results_PCA = np.zeros((0,5))
        results_UMAP = np.zeros((0,5))

        selection_ground = np.random.choice(len(self.known_true), size, replace=False)
        ground = self._get_features(self.known_true[selection_ground])

        reducer_PCA = PCA(n_components = 25)
        ground_PCA = reducer_PCA.fit_transform(ground)
        print("PCA reducer fitted")

        reducer_UMAP = umap.UMAP(n_components = 25)
        ground_UMAP = reducer_UMAP.fit_transform(ground)
        print("UMAP reducer fitted")

        for prop in proportions:
            no = int(prop * size)
            res_prop = np.zeros((runs,5))
            res_prop[:,0] = prop
            res_prop_PCA = np.zeros((runs, 5))
            res_prop_PCA[:, 0] = prop
            res_prop_UMAP = np.zeros((runs,5))
            res_prop_UMAP[:, 0] = prop

            print("Starting proportion: ", prop, "--- true", no)

            for run in range(runs):
                selection_true = np.random.choice(len(self.unknown_true), no, replace=False)
                selection_false = np.random.choice(len(self.unknown_false), (size - no), replace=False)

                if no != 0:
                    new_true = self._get_features(self.unknown_true[selection_true])

                if (size-no) != 0:
                    new_false = self._get_features(self.unknown_false[selection_false])

                if no == 0:
                    new = new_false
                elif (size-no) == 0:
                    new = new_true
                else:
                    new = np.vstack((new_true, new_false))

                # DIMENSIONALITY REDUCTION HERE
                new_PCA = reducer_PCA.transform(new)
                new_UMAP = reducer_UMAP.transform(new)

                res_prop[run, 1] = run + 1
                res_prop_PCA[run, 1] = run + 1
                res_prop_UMAP[run, 1] = run + 1
                res_prop[run, 2:5] = msrs.get_all_scores(ground, new)
                res_prop_PCA[run, 2:5] = msrs.get_all_scores(ground_PCA, new_PCA)
                res_prop_UMAP[run, 2:5] = msrs.get_all_scores(ground_UMAP, new_UMAP)

                print("Run ", run + 1, " completed")

            results = np.vstack((results, res_prop))
            results_PCA = np.vstack((results_PCA, res_prop_PCA))
            results_UMAP = np.vstack((results_UMAP, res_prop_UMAP))

        if pdtable is False:
            return results, results_PCA, results_UMAP

        red_none = np.repeat('none', runs*len(proportions))
        red_PCA = np.repeat('PCA', runs*len(proportions))
        red_UMAP = np.repeat('UMAP', runs*len(proportions))

        wass = np.stack((results[:, 0], results[:, 1], results[:, 2], np.repeat('wasserstein', runs*len(proportions)), red_none), axis = -1)
        kmmd = np.stack((results[:, 0], results[:, 1], results[:, 3], np.repeat('kmmd', runs*len(proportions)), red_none), axis = -1)
        frch = np.stack((results[:, 0], results[:, 1], results[:, 4], np.repeat('frechet', runs*len(proportions)), red_none), axis = -1)

        wass_PCA = np.stack((results_PCA[:, 0], results_PCA[:, 1], results_PCA[:, 2], np.repeat('wasserstein', runs*len(proportions)), red_PCA), axis = -1)
        kmmd_PCA = np.stack((results_PCA[:, 0], results_PCA[:, 1], results_PCA[:, 3], np.repeat('kmmd', runs*len(proportions)), red_PCA), axis = -1)
        frch_PCA = np.stack((results_PCA[:, 0], results_PCA[:, 1], results_PCA[:, 4], np.repeat('frechet', runs*len(proportions)), red_PCA), axis = -1)

        wass_UMAP = np.stack((results_UMAP[:, 0], results_UMAP[:, 1], results_UMAP[:, 2], np.repeat('wasserstein', runs*len(proportions)), red_UMAP), axis = -1)
        kmmd_UMAP = np.stack((results_UMAP[:, 0], results_UMAP[:, 1], results_UMAP[:, 3], np.repeat('kmmd', runs*len(proportions)), red_UMAP), axis = -1)
        frch_UMAP = np.stack((results_UMAP[:, 0], results_UMAP[:, 1], results_UMAP[:, 4], np.repeat('frechet', runs*len(proportions)), red_UMAP), axis = -1)

        total_results = np.concatenate((wass, kmmd, frch, wass_PCA, kmmd_PCA, frch_PCA, wass_UMAP, kmmd_UMAP, frch_UMAP), axis = 0)

        pd_results = pd.DataFrame(total_results, columns = ['proportion', 'run', 'value', 'measure', 'dim reduction'])

        pd_results['value'] = pd_results['value'].astype('float64')
        pd_results['run'] = pd_results['run'].astype('float64')
        pd_results['proportion'] = pd_results['proportion'].astype('category')
        pd_results['measure'] = pd_results['measure'].astype('category')
        pd_results['dim reduction'] = pd_results['dim reduction'].astype('category')

        return pd_results

class SingleObs():
    """
    Performs an experiment investigating the distance measure behaviour when looking at single observations.
    Distance between new observation and specified number of known observation is calculated.
    This new distance is then compared to the reference distribution.
    The reference distribution contains distances between known observations.
    use: init > get_reference_dist > run_exp_distance > run_exp_classif
    """
    def __init__(self, model):

        self.transformer = model
        self.known_true, self.unknown_true, self.unknown_false = self._get_data()

    def _get_data(self):
        """Obtains the data and performs necessary preprocessing"""
        (raw_known_true,_), (raw_unknown_true,_) = keras.datasets.cifar10.load_data()
        (_,_), (raw_unknown_false,_) = keras.datasets.cifar100.load_data()

        known_true = raw_known_true.astype('float32')
        unknown_true = raw_unknown_true.astype('float32')
        unknown_false = raw_unknown_false.astype('float32')

        del raw_known_true, raw_unknown_true, raw_unknown_false

        known_true = known_true / 127.5 - 1
        unknown_true = unknown_true / 127.5 - 1
        unknown_false = unknown_false / 127.5 - 1

        return known_true, unknown_true, unknown_false

    def _get_features(self, subset):
        """Obtains the representation of the observations in the features space of the provided transformer model"""
        features = self.transformer.predict(subset)
        mx_features = np.asarray(features)

        return mx_features
    
    def _get_dim_reducers(self, ref_obs_fts, split, dim_reds = ["PCA"]):
        
        models_PCA = []
        if "PCA" in dim_reds:
            for i in range(len(ref_obs_fts)):
                model = PCA(n_components = split)
                model.fit(np.reshape(ref_obs_fts[i], (split, -1)))
                models_PCA.append(model)
        
        return models_PCA
    
    def _get_reference_dist(self, samples, split, dim_red = [None, "PCA", "UMAP"]):
        """Obtains a reference distribution of the distances for known non-fraudulent observations"""
        ref_av_dist = np.zeros((samples, 4))
        ref_av_dist_PCA = np.zeros((samples, 4))
        #ref_av_dist_UMAP = np.zeros((samples, 4))
        
        print("START - Obtaining reference distribution")
        
        ground_ids = np.random.choice(len(self.known_true_ground), samples, replace = False)
        ground_fts = self.transformer.predict(self.known_true_ground[ground_ids])
        
        for sample in range(samples):
            sample_fts = ground_fts[sample]
            
            sample_distances = np.zeros((samples, 3))
            sample_distances_PCA = np.zeros((samples, 3))
            for ref_obs in range(len(self.ref_obs_fts)):
                main = np.reshape(self.ref_obs_fts[ref_obs], (split, -1))
                comp = np.reshape(sample_fts, (split, -1))
                
                main_PCA = self.reducers_PCA[ref_obs].transform(main)
                comp_PCA = self.reducers_PCA[ref_obs].transform(comp)
                
                sample_distances[ref_obs, :] = msrs.get_all_scores(main, comp)
                sample_distances_PCA[ref_obs, :] = msrs.get_all_scores(main_PCA, comp_PCA)
            
            ref_av_dist[sample, 0:3] = sample_distances.mean(axis = 0)
            ref_av_dist_PCA[sample, 0:3] = sample_distances_PCA.mean(axis = 0)
            
            print("Sample ", (sample + 1), " out of ", samples, "obtained")
        
        ref_av_dist[:, 3] = split
        ref_av_dist_PCA[:, 3]  = split
        
        return ref_av_dist, ref_av_dist_PCA

    def _calc_av_distances(self, observation, ref_obs, split, dim_red):

        observation = observation[None,:,:,:]
        fts_observation = self._get_features(observation)
        ind_dist = np.zeros((ref_obs, 3))
        
        if dim_red == None:
            for obs in range(ref_obs):
                ground = np.reshape(self.ref_obs_fts[obs], (split, -1))
                new = np.reshape(fts_observation, (split, -1))

                ind_dist[obs] = msrs.get_all_scores(ground, new)
            

        elif dim_red =="PCA":
            for obs in range(ref_obs):
                ground = np.reshape(self.ref_obs_fts[obs], (split, -1))
                new = np.reshape(fts_observation, (split, -1))
                
                ground_PCA = self.reducers_PCA[obs].transform(ground)
                new_PCA = self.reducers_PCA[obs].transform(new)
                
                ind_dist[obs] = msrs.get_all_scores(ground_PCA, new_PCA)            
            
        ans = ind_dist.mean(axis = 0)
        
        return ans



    def run_exp_distance(self, ref_obs, dist_size, new_obs, proportion_fake = 0.5, splits = [4, 8, 16], seedval = None):
        """
        Runs the experiment in which the reference distance distribution is obtained, and the distances of new observations are obtained.

        ref_obs -- number of reference observations to which the distances are calculated (and the average taken).
        new_obs -- number of new observations to calculate the average distances to.
        dist_size -- number of known true obervations to calculate the average distances to.
        proportion -- proportion of new observations that is fake.
        split -- number of dimension that 1D feature activation vector is divided into.

        FAKE = 1
        TRUE = 0
        """
        np.random.seed(seed = seedval)

        # CHOOSE REFERENCE OBSERVATIONS
        reference_ids = np.random.choice(int(len(self.known_true)), ref_obs, replace = False)
        self.ref_obs_fts = self._get_features(self.known_true[reference_ids])
        # remove reference observations from the known true 
        self.known_true_ground = np.delete(self.known_true, reference_ids, axis = 0)
        
        references = np.zeros((0, 4))
        references_PCA = np.zeros((0, 4))
        observed_all = np.zeros((0, 5))
        observed_all_PCA = np.zeros((0, 5))
        
        for split in splits:
            print("CURRENT - Feature Vector Split into ", split)
            
            self.reducers_PCA = self._get_dim_reducers(self.ref_obs_fts, split)
            print("PCA and UMAP models created")
            
            reference, reference_PCA = self._get_reference_dist(dist_size, split)
            references = np.vstack((references, reference))
            references_PCA = np.vstack((references_PCA, reference_PCA))
            print("FINISH - Reference distributions obtained")
            no_fake  = abs(new_obs * proportion_fake)
            fake_ids = np.random.choice(int(len(self.unknown_false)), int(no_fake), replace = False)
            true_ids = np.random.choice(int(len(self.unknown_true)), int((new_obs - no_fake)), replace = False)
            true_samples = self.unknown_true[true_ids]
            fake_samples = self.unknown_false[fake_ids]
            new_samples = np.vstack((true_samples, fake_samples))

            observed = np.zeros((new_obs, 5))
            observed_PCA = np.zeros((new_obs, 5))
            print("START - Obtaining observed distributions for new observations")

            for sample in range(new_obs):
                observation = new_samples[sample]
                observed[sample, 0:3] = self._calc_av_distances(observation, ref_obs, split, dim_red = None)
                observed_PCA[sample, 0:3] = self._calc_av_distances(observation, ref_obs, split, dim_red = "PCA")
                print("Sample ", (sample + 1), " out of ", new_obs, "obtained")

            print("FINISH - Observed new distances obtained")
            zeros = np.zeros(int(new_obs) - int(no_fake))
            ones = np.ones(int(no_fake))
            classes = np.append(zeros, ones)

            observed[:, 3] = classes
            observed[:, 4] = split
            observed_PCA[:, 3] = classes
            observed_PCA[:, 4] = split
            observed_all = np.vstack((observed_all, observed))
            observed_all_PCA = np.vstack((observed_all_PCA, observed_PCA))
            
            self.reducers_PCA = []

        return references, observed_all, references_PCA, observed_all_PCA

    def run_exp_classif(self, references, observed_all, thresholds = np.arange(0, 1.05, 0.05), pdtable = True):
        """
        Calcutes the classification performance of the threshold based outlier detector.

        reference -- array of reference distances
        observed -- array of new distances and their true classifications (col 3)
        threshold -- percentile of the reference distribtuion above which new obs are classified as fake
        """

        assert (np.unique(references[:, 3]) == np.unique(observed_all[:, 4])).all()
        splits = np.unique(references[:, 3])
        print(splits)


        classifications_all = np.zeros((0, 6))
        for threshold in thresholds:
            print("Currently threshold: ", threshold)
            for split in splits:
                print("Currently split: ", int(split))
                reference = references[references[:, 3] == split, :]
                observed = observed_all[observed_all[:, 4] == split, :]
                
                combined = np.vstack((reference[:, 0:3], observed[:, 0:3]))
                
                emd = np.percentile(combined[:, 0], threshold)
                kmmd = np.percentile(combined[:, 1], threshold)
                frech = np.percentile(combined[:, 2], threshold)

                classifications = np.zeros((len(observed), 6))

                for obs in range(len(observed)):
                    if observed[obs, 0] > emd:
                        classifications[obs, 0] = 1
                    if observed[obs, 1] > kmmd:
                        classifications[obs, 1] = 1
                    if observed[obs, 2] > frech:
                        classifications[obs, 2] = 1
                    classifications[obs, 3] = observed[obs, 3]

                classifications[:, 4] = split
                classifications[:, 5] = threshold
                classifications_all = np.vstack((classifications_all, classifications))

        if pdtable == False:
            return classifications_all

        classifications_all = pd.DataFrame(classifications_all, columns = ["wasserstein", "kMMD", "Frechet", "Truth", "Split", "Threshold"])

        return classifications_all
