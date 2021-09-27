# -*- coding: utf-8 -*-
"""
This is the main script of the MNIST investigation.

It includes:
    - Training of the GAN models.
    - Running of expereiments.
    - Processing and visualising the results.

All the ncessary classes and functions are in the supporting scripts:
> minst_gans.py, mnist_experiments.py, mnist_plots.py
"""
import mnist_gans as gans
import mnist_experiments as exp
import mnist_plots as rd_plt
import pandas as pd
import numpy as np
import time
import os

# Initial set-up
start_directory = os.getcwd()
date_code = "m0d0"
epochs_dc = 300
epochs_dc_code = str(epochs_dc) + "_passes"
epochs_ls = 300
epochs_ls_code = str(epochs_ls) + "_passes"
epochs_w = 300
epochs_w_code = str(epochs_w) + "_passes"

# TRAIN or NOT
dc_train = True
ls_train = True
w_train = True


### DCGAN START ### ###########################################################

## ADMIN: NAME ATTEMPT !!!! ##
os.chdir(start_directory)
attempt_name = "mnist_"+ date_code + "_dcgan_" + epochs_dc_code
final_directory = os.path.join(start_directory, attempt_name)
if not os.path.exists(final_directory):
   os.mkdir(final_directory)
os.chdir(final_directory)

## MODEL TRAINING ##
# initialise gan model
dcgan = gans.dcGanMnist()
if dc_train == True:
    # train gan model
    dcgan.train(epochs = epochs_dc)
    # obtain transformer
    dcgan_transformer = dcgan.get_feature_encoder()
    # save
    dcgan.save_feature_encoder(attempt_name)
    # Plot losses
    rd_plt.make_loss_plots("DCGAN", attempt_name, dcgan.losses, epochs_dc)
    np.save(attempt_name + "_losses", dcgan.losses)
    # Create examples of the generator output
    rd_plt.make_gan_examples(dcgan, 'dcgan')
elif dc_train == False:
    dcgan_transformer = dcgan.load_feature_encoder(attempt_name)

## EXPERIMENT ##
# initialise experiment
dcgan_experiment = exp.ExperimentsMnist(model = dcgan_transformer, random_seed = 5073)

# obtain results
start = time.time()
dcgan_wass = dcgan_experiment.run_wasserstein(size = 100, pixel_space = True)
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
dcgan_kmmd = dcgan_experiment.run_kmmd(size = 100, pixel_space = True)
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
dcgan_frch = dcgan_experiment.run_frechet(size = 100, pixel_space = True)
print("completed in " + str(time.time() - start) + " seconds")

# obtain results
# WITH PCA DIMENSIONALITY REDUCTION
start = time.time()
dcgan_wass_pca = dcgan_experiment.run_wasserstein(size = 100, pixel_space = True, dim_reduction = "PCA")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
dcgan_kmmd_pca = dcgan_experiment.run_kmmd(size = 100, pixel_space = True, dim_reduction = "PCA")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
dcgan_frch_pca = dcgan_experiment.run_frechet(size = 100, pixel_space = True, dim_reduction = "PCA")
print("completed in " + str(time.time() - start) + " seconds")

# obtain results
# WITH UMAP DIMENSIONALITY REDUCTION
start = time.time()
dcgan_wass_umap = dcgan_experiment.run_wasserstein(size = 100, pixel_space = True, dim_reduction = "UMAP")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
dcgan_kmmd_umap = dcgan_experiment.run_kmmd(size = 100, pixel_space = True, dim_reduction = "UMAP")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
dcgan_frch_umap = dcgan_experiment.run_frechet(size = 100, pixel_space = True, dim_reduction = "UMAP")
print("completed in " + str(time.time() - start) + " seconds")

# Save the results
dcgan_wass.to_pickle("results_wass.pkl")
dcgan_kmmd.to_pickle("results_kmmd.pkl")
dcgan_frch.to_pickle("results_frch.pkl")
dcgan_wass_pca.to_pickle("results_wass_pca.pkl")
dcgan_kmmd_pca.to_pickle("results_kmmd_pca.pkl")
dcgan_frch_pca.to_pickle("results_frch_pca.pkl")
dcgan_wass_umap.to_pickle("results_wass_umap.pkl")
dcgan_kmmd_umap.to_pickle("results_kmmd_umap.pkl")
dcgan_frch_umap.to_pickle("results_frch_umap.pkl")

# Create and save plots of the results
rd_plt.make_results_plots("DCGAN", attempt_name, dcgan_wass, dcgan_kmmd, dcgan_frch, dim_red = None, pixel_space = True)
rd_plt.make_results_plots("DCGAN", attempt_name, dcgan_wass_pca, dcgan_kmmd_pca, dcgan_frch_pca, dim_red = "PCA", pixel_space = True)
rd_plt.make_results_plots("DCGAN", attempt_name, dcgan_wass_umap, dcgan_kmmd_umap, dcgan_frch_umap, dim_red = "UMAP", pixel_space = True)

### LSGAN START ### ###########################################################

# naming of the attempt
os.chdir(start_directory)
attempt_name = "mnist_" + date_code + "_lsgan_" + epochs_ls_code
final_directory = os.path.join(start_directory, attempt_name)
if not os.path.exists(final_directory):
   os.mkdir(final_directory)
os.chdir(final_directory)

## MODEL TRAINING ##
lsgan = gans.lsGanMnist()
if ls_train == True:
    # train gan model
    lsgan.train(epochs = epochs_ls)
    # obtain transformer
    lsgan_transformer = lsgan.get_feature_encoder()
    # save
    lsgan.save_feature_encoder(attempt_name)
    # Plot losses
    rd_plt.make_loss_plots("LSGAN", attempt_name, lsgan.losses, epochs_ls)
    np.save(attempt_name + "_losses", lsgan.losses)
    # Create examples of the generator output
    rd_plt.make_gan_examples(lsgan, 'lsgan')
elif dc_train == False:
    lsgan_transformer = lsgan.load_feature_encoder(attempt_name)

## EXPERIMENT ##
# initialise experiment
lsgan_experiment = exp.ExperimentsMnist(model = lsgan_transformer, random_seed = 5073)

# obtain results
start = time.time()
lsgan_wass = lsgan_experiment.run_wasserstein(size = 100, pixel_space = False)
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
lsgan_kmmd = lsgan_experiment.run_kmmd(size = 100, pixel_space = False)
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
lsgan_frch = lsgan_experiment.run_frechet(size = 100, pixel_space = False)
print("completed in " + str(time.time() - start) + " seconds")

# obtain results
# WITH PCA DIMENSIONALITY REDUCTION
start = time.time()
lsgan_wass_pca = lsgan_experiment.run_wasserstein(size = 100, pixel_space = False, dim_reduction = "PCA")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
lsgan_kmmd_pca = lsgan_experiment.run_kmmd(size = 100, pixel_space = False, dim_reduction = "PCA")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
lsgan_frch_pca = lsgan_experiment.run_frechet(size = 100, pixel_space = False, dim_reduction = "PCA")
print("completed in " + str(time.time() - start) + " seconds")

# obtain results
# WITH UMAP DIMENSIONALITY REDUCTION
start = time.time()
lsgan_wass_umap = lsgan_experiment.run_wasserstein(size = 100, pixel_space = False, dim_reduction = "UMAP")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
lsgan_kmmd_umap = lsgan_experiment.run_kmmd(size = 100, pixel_space = False, dim_reduction = "UMAP")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
lsgan_frch_umap = lsgan_experiment.run_frechet(size = 100, pixel_space = False, dim_reduction = "UMAP")
print("completed in " + str(time.time() - start) + " seconds")

# Save the results
lsgan_wass.to_pickle("results_wass.pkl")
lsgan_kmmd.to_pickle("results_kmmd.pkl")
lsgan_frch.to_pickle("results_frch.pkl")
lsgan_wass_pca.to_pickle("results_wass_pca.pkl")
lsgan_kmmd_pca.to_pickle("results_kmmd_pca.pkl")
lsgan_frch_pca.to_pickle("results_frch_pca.pkl")
lsgan_wass_umap.to_pickle("results_wass_umap.pkl")
lsgan_kmmd_umap.to_pickle("results_kmmd_umap.pkl")
lsgan_frch_umap.to_pickle("results_frch_umap.pkl")

# Create and save plots of the results
rd_plt.make_results_plots("DCGAN", attempt_name, lsgan_wass, lsgan_kmmd, lsgan_frch, dim_red = None)
rd_plt.make_results_plots("DCGAN", attempt_name, lsgan_wass_pca, lsgan_kmmd_pca, lsgan_frch_pca, dim_red = "PCA")
rd_plt.make_results_plots("DCGAN", attempt_name, lsgan_wass_umap, lsgan_kmmd_umap, lsgan_frch_umap, dim_red = "UMAP")


### WGAN START ### ###########################################################

# naming of the attempt
os.chdir(start_directory)
attempt_name = "mnist_" + date_code + "_wgan_" + epochs_w_code
final_directory = os.path.join(start_directory, attempt_name)
if not os.path.exists(final_directory):
   os.mkdir(final_directory)
os.chdir(final_directory)

## MODEL TRAINING ##
wgan = gans.wGanMnist()
if w_train == True:
    # train gan model
    wgan.train(epochs = epochs_w)
    # obtain transformer
    wgan_transformer = wgan.get_feature_encoder()
    # save
    wgan.save_feature_encoder(attempt_name)
    # Plot losses
    rd_plt.make_loss_plots("WGAN", attempt_name, wgan.losses, epochs_w)
    np.save(attempt_name + "_losses", wgan.losses)
    # Create examples of the generator output
    rd_plt.make_gan_examples(wgan, 'wgan')
elif dc_train == False:
    wgan_transformer = wgan.load_feature_encoder(attempt_name)

## EXPERIMENT ##
# initialise experiment
wgan_experiment = exp.ExperimentsMnist(model = wgan_transformer, random_seed = 5073)

# obtain results
start = time.time()
wgan_wass = wgan_experiment.run_wasserstein(size = 100, pixel_space = False)
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
wgan_kmmd = wgan_experiment.run_kmmd(size = 100, pixel_space = False)
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
wgan_frch = wgan_experiment.run_frechet(size = 100, pixel_space = False)
print("completed in " + str(time.time() - start) + " seconds")

# obtain results
# WITH PCA DIMENSIONALITY REDUCTION
start = time.time()
wgan_wass_pca = wgan_experiment.run_wasserstein(size = 100, pixel_space = False, dim_reduction = "PCA")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
wgan_kmmd_pca = wgan_experiment.run_kmmd(size = 100, pixel_space = False, dim_reduction = "PCA")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
wgan_frch_pca = wgan_experiment.run_frechet(size = 100, pixel_space = False, dim_reduction = "PCA")
print("completed in " + str(time.time() - start) + " seconds")

# obtain results
# WITH UMAP DIMENSIONALITY REDUCTION
start = time.time()
wgan_wass_umap = wgan_experiment.run_wasserstein(size = 100, pixel_space = False, dim_reduction = "UMAP")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
wgan_kmmd_umap = wgan_experiment.run_kmmd(size = 100, pixel_space = False, dim_reduction = "UMAP")
print("completed in " + str(time.time() - start) + " seconds")

start = time.time()
wgan_frch_umap = wgan_experiment.run_frechet(size = 100, pixel_space = False, dim_reduction = "UMAP")
print("completed in " + str(time.time() - start) + " seconds")

# Save the results
wgan_wass.to_pickle("results_wass.pkl")
wgan_kmmd.to_pickle("results_kmmd.pkl")
wgan_frch.to_pickle("results_frch.pkl")
wgan_wass_pca.to_pickle("results_wass_pca.pkl")
wgan_kmmd_pca.to_pickle("results_kmmd_pca.pkl")
wgan_frch_pca.to_pickle("results_frch_pca.pkl")
wgan_wass_umap.to_pickle("results_wass_umap.pkl")
wgan_kmmd_umap.to_pickle("results_kmmd_umap.pkl")
wgan_frch_umap.to_pickle("results_frch_umap.pkl")

# Create and save plots of the results
rd_plt.make_results_plots("DCGAN", attempt_name, wgan_wass, wgan_kmmd, wgan_frch, dim_red = None)
rd_plt.make_results_plots("DCGAN", attempt_name, wgan_wass_pca, wgan_kmmd_pca, wgan_frch_pca, dim_red = "PCA")
rd_plt.make_results_plots("DCGAN", attempt_name, wgan_wass_umap, wgan_kmmd_umap, wgan_frch_umap, dim_red = "UMAP")

