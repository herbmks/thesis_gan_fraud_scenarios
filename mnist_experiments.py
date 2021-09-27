# -*- coding: utf-8 -*-
"""
This script inludes the MNIST experiment.
The experiments aims to show the difference between performing the distance
    calculations in the pixel space and the feature space.
For the feature representation, we use the feature space of GAN discriminators.
    These GANs and their feature spaces are accessible fron a seperate script.
> MNIST_gans.py
For the distance measures, we use those accessible from a different script.
> measures.py
"""

from tensorflow.keras.datasets import mnist

import numpy as np
#import MNIST_gans as gans
import measures as msrs
import pandas as pd
from sklearn.decomposition import PCA
import umap


class ExperimentsMnist():
    """Performs and experiment calculating the distance between known and unknown, shuffled, and random data"""

    def __init__(self, model, random_seed = None):
        self.known, self.unknown, self.shuffled, self.random = self._get_raw_data(random_seed)
        self.known_pxl, self.unknown_pxl, self.shuffled_pxl, self.random_pxl = self._get_pxl_data()
        self.known_fts, self.unknown_fts, self.shuffled_fts, self.random_fts = self._get_feature_data(transformer=model)

        del self.known, self.unknown, self.shuffled, self.random

    def _get_raw_data(self, random_seed):
        """Retrieves the raw data"""
        (x_train,_), (x_test,_) = mnist.load_data()
        np.random.seed(seed = random_seed)
        np.random.shuffle(x_train)
        np.random.shuffle(x_test)
        x_train = x_train / 127.5 - 1
        x_test = x_test / 127.5 - 1
        x_train = np.expand_dims(x_train, axis = 3)
        x_test = np.expand_dims(x_test, axis = 3)
        known = x_train[0:5000]
        unknown = x_test[0:5000]
        shuffled = x_test[5000:10000]
        random = (np.random.rand(5000,28,28,1) * 2) -1

        for i in range(len(shuffled)):
            temp = np.asarray(shuffled[i])
            temp = temp.flatten()
            np.random.shuffle(temp)
            temp = temp.reshape(28,28)
            np.append(shuffled, temp)

        del x_train, x_test

        return known, unknown, shuffled, random

    def _get_feature_data(self, transformer):
        """Obtain data in feature space of given transformer"""


        known_fts = transformer.predict(self.known)
        unknown_fts = transformer.predict(self.unknown)
        shuffled_fts = transformer.predict(self.shuffled)
        random_fts = transformer.predict(self.random)

        mx_known_fts = np.asarray(known_fts)
        mx_unknown_fts = np.asarray(unknown_fts)
        mx_shuffled_fts = np.asarray(shuffled_fts)
        mx_random_fts = np.asarray(random_fts)

        return mx_known_fts, mx_unknown_fts, mx_shuffled_fts, mx_random_fts

    def _get_pxl_data(self):
        """Obtain data in the pixel space"""
        known_pxl = np.reshape(self.known, (5000, 28*28))
        unknown_pxl = np.reshape(self.unknown, (5000, 28*28))
        shuffled_pxl = np.reshape(self.shuffled, (5000, 28*28))
        random_pxl = np.reshape(self.random, (5000, 28*28))

        return known_pxl, unknown_pxl, shuffled_pxl, random_pxl

    def _reduce_dims(self, reference, target, method = None):
        """
        Reduces the dimensionality of the data.
        Method: PCA or UMAP or None
        """
        if method == None:
            return reference, target
        
        if method == "PCA":
            reducer = PCA(n_components = 25)
            Xred = reducer.fit_transform(reference)
            Yred = reducer.transform(target)
            return Xred, Yred

        elif method == "UMAP":
            reducer = umap.UMAP()
            Xred = reducer.fit_transform(reference)
            Yred = reducer.transform(target)
            return Xred, Yred

        else:
            print("ERROR: method must be one of None, PCA, UMAP")

    def run_wasserstein(self, size = 100, dim_reduction = None, pixel_space = False, pdtable = True):
        """
        Performs the experiment using the Wasserstein metric.
        size: size of samples that are compared
        dim_reduction: One of (PCA, UMAP, None)
        """
        runs = len(self.known_pxl)//size
        pxl_distances = np.zeros((runs, 3))
        fts_distances = np.zeros((runs, 3))

        for i in range(runs):
            # have to import measures
            if pixel_space == True:
                kn1, un1 = self._reduce_dims(self.known_pxl[size*i:size*(i+1)], self.unknown_pxl[size*i:size*(i+1)], method = dim_reduction)
                pxl_distances[i, 0] = msrs.get_wasserstein(kn1, un1)
                kn2, un2 = self._reduce_dims(self.known_pxl[size*i:size*(i+1)], self.shuffled_pxl[size*i:size*(i+1)], method = dim_reduction)
                pxl_distances[i, 1] = msrs.get_wasserstein(kn2, un2)
                kn3, un3 = self._reduce_dims(self.known_pxl[size*i:size*(i+1)], self.random_pxl[size*i:size*(i+1)], method = dim_reduction)
                pxl_distances[i, 2] = msrs.get_wasserstein(kn3, un3)
            kn4, un4 = self._reduce_dims(self.known_fts[size*i:size*(i+1)], self.unknown_fts[size*i:size*(i+1)], method = dim_reduction)
            fts_distances[i, 0] = msrs.get_wasserstein(kn4, un4)
            kn5, un5 = self._reduce_dims(self.known_fts[size*i:size*(i+1)], self.shuffled_fts[size*i:size*(i+1)], method = dim_reduction)
            fts_distances[i, 1] = msrs.get_wasserstein(kn5, un5)
            kn6, un6 = self._reduce_dims(self.known_fts[size*i:size*(i+1)], self.random_fts[size*i:size*(i+1)], method = dim_reduction)
            fts_distances[i, 2] = msrs.get_wasserstein(kn6, un6)

            if ((i+1) % 10) == 0:
                print("Wasserstein run " + str(i+1) + " of " + str(runs) + " completed")

        if pdtable is False:
            return pxl_distances, fts_distances

        if pixel_space == True:
            feature = np.repeat('feature', runs*3)
            pixel = np.repeat('pixel', runs*3)
            unknown = np.repeat('unknown', runs)
            shuffled = np.repeat('shuffled', runs)
            random = np.repeat('random', runs)
            measure = np.repeat('wasserstein', runs*6)

            if dim_reduction == None:
                red_method = np.repeat('none', runs*6)
            else:
                red_method = np.repeat(dim_reduction, runs*6)

            dist = np.hstack((pxl_distances[:, 0], pxl_distances[:, 1], pxl_distances[:, 2],
                          fts_distances[:, 0], fts_distances[:, 1], fts_distances[:, 2]))
            space = np.hstack((pixel, feature))
            data = np.hstack((unknown, shuffled, random, unknown, shuffled, random))

            results = np.stack((dist, space, data, measure, red_method), axis = -1)
            results = pd.DataFrame(results, columns = ['value', 'space', 'dataset', 'measure', 'dim reduction'])

            results['value'] = results['value'].astype('float64')
            results['space'] = results['space'].astype('category')
            results['dataset'] = results['dataset'].astype('category')
            results['measure'] = results['measure'].astype('category')

        elif pixel_space == False:
            feature = np.repeat('feature', runs*3)
            unknown = np.repeat('unknown', runs)
            shuffled = np.repeat('shuffled', runs)
            random = np.repeat('random', runs)
            measure = np.repeat('wasserstein', runs*3)

            if dim_reduction == None:
                red_method = np.repeat('none', runs*3)
            else:
                red_method = np.repeat(dim_reduction, runs*3)

            dist = np.hstack((fts_distances[:, 0], fts_distances[:, 1], fts_distances[:, 2]))
            space = np.hstack((feature))
            data = np.hstack((unknown, shuffled, random))

            results = np.stack((dist, space, data, measure, red_method), axis = -1)
            results = pd.DataFrame(results, columns = ['value', 'space', 'dataset', 'measure', 'dim reduction'])

            results['value'] = results['value'].astype('float64')
            results['space'] = results['space'].astype('category')
            results['dataset'] = results['dataset'].astype('category')
            results['measure'] = results['measure'].astype('category')

        return results

    def run_kmmd(self, size = 100, dim_reduction = None, pixel_space = False, pdtable = True):
        """
        Performs the experiment using Kernel MMD metric
        size: size of samples that are compared
        dim_reduction: One of (PCA, UMAP, None)
        """
        runs = abs(len(self.known_pxl)//size)
        pxl_distances = np.zeros((runs,3))
        fts_distances = np.zeros((runs,3))

        for i in range(runs):
            if pixel_space == True:
                kn1, un1 = self._reduce_dims(self.known_pxl[size*i:size*(i+1)], self.unknown_pxl[size*i:size*(i+1)], method = dim_reduction)
                pxl_distances[i, 0] = msrs.get_kmmd(kn1, un1)
                kn2, un2 = self._reduce_dims(self.known_pxl[size*i:size*(i+1)], self.shuffled_pxl[size*i:size*(i+1)], method = dim_reduction)
                pxl_distances[i, 1] = msrs.get_kmmd(kn2, un2)
                kn3, un3 = self._reduce_dims(self.known_pxl[size*i:size*(i+1)], self.random_pxl[size*i:size*(i+1)], method = dim_reduction)
                pxl_distances[i, 2] = msrs.get_kmmd(kn3, un3)
            kn4, un4 = self._reduce_dims(self.known_fts[size*i:size*(i+1)], self.unknown_fts[size*i:size*(i+1)], method = dim_reduction)
            fts_distances[i, 0] = msrs.get_kmmd(kn4, un4)
            kn5, un5 = self._reduce_dims(self.known_fts[size*i:size*(i+1)], self.shuffled_fts[size*i:size*(i+1)], method = dim_reduction)
            fts_distances[i, 1] = msrs.get_kmmd(kn5, un5)
            kn6, un6 = self._reduce_dims(self.known_fts[size*i:size*(i+1)], self.random_fts[size*i:size*(i+1)], method = dim_reduction)
            fts_distances[i, 2] = msrs.get_kmmd(kn6, un6)

            if ((i+1) % 10) == 0:
                print("Kmmd run " + str(i+1) + " of " + str(runs) + " completed")

        if pdtable is False:
            return pxl_distances, fts_distances

        if pixel_space == True:
            feature = np.repeat('feature', runs*3)
            pixel = np.repeat('pixel', runs*3)
            unknown = np.repeat('unknown', runs)
            shuffled = np.repeat('shuffled', runs)
            random = np.repeat('random', runs)
            measure = np.repeat('kernel mmd', runs*6)

            if dim_reduction == None:
                red_method = np.repeat('none', runs*6)
            else:
                red_method = np.repeat(dim_reduction, runs*6)

            dist = np.hstack((pxl_distances[:, 0], pxl_distances[:, 1], pxl_distances[:, 2],
                          fts_distances[:, 0], fts_distances[:, 1], fts_distances[:, 2]))
            space = np.hstack((pixel, feature))
            data = np.hstack((unknown, shuffled, random, unknown, shuffled, random))

            results = np.stack((dist, space, data, measure, red_method), axis = -1)
            results = pd.DataFrame(results, columns = ['value', 'space', 'dataset', 'measure', 'dim reduction'])

            results['value'] = results['value'].astype('float64')
            results['space'] = results['space'].astype('category')
            results['dataset'] = results['dataset'].astype('category')
            results['measure'] = results['measure'].astype('category')

        elif pixel_space == False:
            feature = np.repeat('feature', runs*3)
            unknown = np.repeat('unknown', runs)
            shuffled = np.repeat('shuffled', runs)
            random = np.repeat('random', runs)
            measure = np.repeat('kernel mmd', runs*3)

            if dim_reduction == None:
                red_method = np.repeat('none', runs*3)
            else:
                red_method = np.repeat(dim_reduction, runs*3)

            dist = np.hstack((fts_distances[:, 0], fts_distances[:, 1], fts_distances[:, 2]))
            space = np.hstack((feature))
            data = np.hstack((unknown, shuffled, random))

            results = np.stack((dist, space, data, measure, red_method), axis = -1)
            results = pd.DataFrame(results, columns = ['value', 'space', 'dataset', 'measure', 'dim reduction'])

            results['value'] = results['value'].astype('float64')
            results['space'] = results['space'].astype('category')
            results['dataset'] = results['dataset'].astype('category')
            results['measure'] = results['measure'].astype('category')

        return results

    def run_frechet(self, size = 100, dim_reduction = None, pixel_space = False, pdtable = True):
        """
        Performs the experimentusing the Frechet metric
        size: size of samples that are compared
        dim_reduction: One of (PCA, UMAP, None)
        """
        runs = abs(len(self.known_pxl)//size)
        pxl_distances = np.zeros((runs,3))
        fts_distances = np.zeros((runs,3))

        for i in range(runs):
            if pixel_space == True:
                kn1, un1 = self._reduce_dims(self.known_pxl[size*i:size*(i+1)], self.unknown_pxl[size*i:size*(i+1)], method = dim_reduction)
                pxl_distances[i, 0] = msrs.frechet(kn1, un1)
                kn2, un2 = self._reduce_dims(self.known_pxl[size*i:size*(i+1)], self.shuffled_pxl[size*i:size*(i+1)], method = dim_reduction)
                pxl_distances[i, 1] = msrs.frechet(kn2, un2)
                kn3, un3 = self._reduce_dims(self.known_pxl[size*i:size*(i+1)], self.random_pxl[size*i:size*(i+1)], method = dim_reduction)
                pxl_distances[i, 2] = msrs.frechet(kn3, un3)
            kn4, un4 = self._reduce_dims(self.known_fts[size*i:size*(i+1)], self.unknown_fts[size*i:size*(i+1)], method = dim_reduction)
            fts_distances[i, 0] = msrs.frechet(kn4, un4)
            kn5, un5 = self._reduce_dims(self.known_fts[size*i:size*(i+1)], self.shuffled_fts[size*i:size*(i+1)], method = dim_reduction)
            fts_distances[i, 1] = msrs.frechet(kn5, un5)
            kn6, un6 = self._reduce_dims(self.known_fts[size*i:size*(i+1)], self.random_fts[size*i:size*(i+1)], method = dim_reduction)
            fts_distances[i, 2] = msrs.frechet(kn6, un6)

            print("Frechet run " + str(i+1) + " of " + str(runs) + " completed")

        if pdtable is False:
            return pxl_distances, fts_distances

        if pixel_space == True:
            feature = np.repeat('feature', runs*3)
            pixel = np.repeat('pixel', runs*3)
            unknown = np.repeat('unknown', runs)
            shuffled = np.repeat('shuffled', runs)
            random = np.repeat('random', runs)
            measure = np.repeat('frechet', runs*6)

            if dim_reduction == None:
                red_method = np.repeat('none', runs*6)
            else:
                red_method = np.repeat(dim_reduction, runs*6)

            dist = np.hstack((pxl_distances[:, 0], pxl_distances[:, 1], pxl_distances[:, 2],
                          fts_distances[:, 0], fts_distances[:, 1], fts_distances[:, 2]))
            space = np.hstack((pixel, feature))
            data = np.hstack((unknown, shuffled, random, unknown, shuffled, random))

            results = np.stack((dist, space, data, measure, red_method), axis = -1)
            results = pd.DataFrame(results, columns = ['value', 'space', 'dataset', 'measure', 'dim reduction'])

            results['value'] = results['value'].astype('float64')
            results['space'] = results['space'].astype('category')
            results['dataset'] = results['dataset'].astype('category')
            results['measure'] = results['measure'].astype('category')

        elif pixel_space == False:
            feature = np.repeat('feature', runs*3)
            unknown = np.repeat('unknown', runs)
            shuffled = np.repeat('shuffled', runs)
            random = np.repeat('random', runs)
            measure = np.repeat('frechet', runs*3)

            if dim_reduction == None:
                red_method = np.repeat('none', runs*3)
            else:
                red_method = np.repeat(dim_reduction, runs*3)

            dist = np.hstack((fts_distances[:, 0], fts_distances[:, 1], fts_distances[:, 2]))
            space = np.hstack((feature))
            data = np.hstack((unknown, shuffled, random))

            results = np.stack((dist, space, data, measure, red_method), axis = -1)
            results = pd.DataFrame(results, columns = ['value', 'space', 'dataset', 'measure', 'dim reduction'])

            results['value'] = results['value'].astype('float64')
            results['space'] = results['space'].astype('category')
            results['dataset'] = results['dataset'].astype('category')
            results['measure'] = results['measure'].astype('category')

        return results

