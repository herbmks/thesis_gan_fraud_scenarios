# -*- coding: utf-8 -*-
"""
This is the main script of the CIFAR investigation.

It includes:
    - Training of the GAN models.
    - Running of expereiments.
    - Processing and visualising the results.

All the ncessary classes and functions are in the supporting scripts:
> cifar_models.py, cifar_experiments.py, cifar_plots.py, cifar_utils.py
"""
import cifar_models as mdls
import cifar_experiments as exp
import cifar_utils as utls
import cifar_plots as rd_plt
import pandas as pd
import time
import os
import numpy as np

# Initial set-up
start_directory = os.getcwd()
date_code = "m0d0"
epochs_dc = 500
epochs_ls = 80

# TRAIN or NOT
dc_train = False
ls_train = False
w_train = False

### DCGAN START ### ###########################################################
date_code = "m08d07"


## ADMIN: NAME ATTEMPT !!!! ##
os.chdir(start_directory)
attempt_name = "cifar_"+ date_code + "_dcgan_" + str(epochs_dc) + "_passes"
final_directory = os.path.join(start_directory, attempt_name)
if not os.path.exists(final_directory):
   os.mkdir(final_directory)
os.chdir(final_directory)

dcgan_name = "DCGAN"
dcgan_code = "DCGAN"

## MODEL TRAINING ##
# initialise gan model
dcgan = mdls.dcGanCifar10()
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

date_code = "m08d12"

os.chdir(start_directory)
attempt_name = "cifar_"+ date_code + "_dcgan_" + str(epochs_dc) + "_passes"
final_directory = os.path.join(start_directory, attempt_name)
if not os.path.exists(final_directory):
   os.mkdir(final_directory)
os.chdir(final_directory)
 
## EXPERIMENT 1 - MULTIPLE OBS ##

dcgan_exp1 = exp.MultipleObs(dcgan_transformer)

dcgan_exp1_res500 = dcgan_exp1.run_experiment(size = 500, runs = 100, proportions = [0, 0.2, 0.4, 0.6, 0.8, 1], seedval = 5073)
dcgan_exp1_res500.to_csv('exp1_results_n500.csv', index=False)
rd_plt.plot1_prop_res(modelname = dcgan_name, modelcode = dcgan_code, full_results = dcgan_exp1_res500, n = 500)

dcgan_exp1_res250 = dcgan_exp1.run_experiment(size = 250, runs = 100, proportions = [0, 0.2, 0.4, 0.6, 0.8, 1], seedval = 5073)
dcgan_exp1_res250.to_csv('exp1_results_n250.csv', index=False)
rd_plt.plot1_prop_res(modelname = dcgan_name, modelcode = dcgan_code, full_results = dcgan_exp1_res250, n = 250)

dcgan_exp1_res100 = dcgan_exp1.run_experiment(size = 100, runs = 100, proportions = [0, 0.2, 0.4, 0.6, 0.8, 1], seedval = 5073)
dcgan_exp1_res100.to_csv('exp1_results_n100.csv', index=False)
rd_plt.plot1_prop_res(modelname = dcgan_name, modelcode = dcgan_code, full_results = dcgan_exp1_res100, n = 100)

'''
## EXPERIMENT 2 - SINGLE OBS ##
dcgan_exp2 = exp.SingleObs(dcgan_transformer)
dcgan_exp2_ref, dcgan_exp2_obs, dcgan_exp2_ref_PCA, dcgan_exp2_obs_PCA  = dcgan_exp2.run_exp_distance(ref_obs = 250, dist_size = 1000, new_obs = 100, seedval = 5073)

np.save('dcgan_exp2_ref.npy', dcgan_exp2_ref)
np.save('dcgan_exp2_obs.npy', dcgan_exp2_obs)
np.save('dcgan_exp2_ref_PCA.npy', dcgan_exp2_ref_PCA)
np.save('dcgan_exp2_obs_PCA.npy', dcgan_exp2_obs_PCA)
   

dcgan_exp2_class = dcgan_exp2.run_exp_classif(dcgan_exp2_ref, dcgan_exp2_obs)
dcgan_exp2_class_PCA = dcgan_exp2.run_exp_classif(dcgan_exp2_ref_PCA, dcgan_exp2_obs_PCA)

dcgan_exp2_conf = utls.get_basic_confusion_vals(dcgan_exp2_class)
dcgan_exp2_conf_long = utls.convert_confusion_to_long(dcgan_exp2_conf)
dcgan_exp2_conf_PCA = utls.get_basic_confusion_vals(dcgan_exp2_class_PCA)
dcgan_exp2_conf_long_PCA = utls.convert_confusion_to_long(dcgan_exp2_conf_PCA)

dcgan_exp2_class.to_csv('exp2_class.csv', index = False)
dcgan_exp2_conf.to_csv('exp2_conf.csv', index = False)
dcgan_exp2_class_PCA.to_csv('exp2_class_PCA.csv', index = False)
dcgan_exp2_conf_PCA.to_csv('exp2_conf_PCA.csv', index = False)

rd_plt.plot2_classification(dcgan_name, dcgan_code, dcgan_exp2_conf_long, dim_red = None)
rd_plt.plot2_classification(dcgan_name, dcgan_code, dcgan_exp2_conf_long_PCA, dim_red = "PCA")

dcgan_exp2_conf_adv = utls.get_adv_confusion_vals(dcgan_exp2_conf)
dcgan_exp2_conf_adv_PCA = utls.get_adv_confusion_vals(dcgan_exp2_conf_PCA)

rd_plt.plot2_roc_pr(dcgan_name, dcgan_code, dcgan_exp2_conf_adv, dim_red = None)
rd_plt.plot2_roc_pr(dcgan_name, dcgan_code, dcgan_exp2_conf_adv_PCA, dim_red = "PCA")
'''

### LSGAN START ### ###########################################################


### INCEPTION START ### ###########################################################
date_code = "m08d12"

## ADMIN: NAME ATTEMPT !!!! ##
os.chdir(start_directory)
attempt_name = "cifar_"+ date_code + "_inception"
final_directory = os.path.join(start_directory, attempt_name)
if not os.path.exists(final_directory):
   os.mkdir(final_directory)
os.chdir(final_directory)

incep_name = "Inception"
incep_code = "inceptionV3"

## OBTAINING ENCODER ##
inception = mdls.inception()
inception_transformer = inception.get_feature_encoder()


## EXPERIMENT 1 - MULTIPLE OBS ##

incep_exp1 = exp.MultipleObs(inception_transformer)

incep_exp1_res500 = incep_exp1.run_experiment(size = 500, runs = 100, proportions = [0, 0.2, 0.4, 0.6, 0.8, 1], seedval = 5073)
incep_exp1_res500.to_csv('exp1_results_n500.csv', index=False)
rd_plt.plot1_prop_res(modelname = incep_name, modelcode = incep_code, full_results = incep_exp1_res500, n = 500)

incep_exp1_res250 = incep_exp1.run_experiment(size = 250, runs = 100, proportions = [0, 0.2, 0.4, 0.6, 0.8, 1], seedval = 5073)
incep_exp1_res250.to_csv('exp1_results_n250.csv', index=False)
rd_plt.plot1_prop_res(modelname = incep_name, modelcode = incep_code, full_results = incep_exp1_res250, n = 250)

incep_exp1_res100 = incep_exp1.run_experiment(size = 100, runs = 100, proportions = [0, 0.2, 0.4, 0.6, 0.8, 1], seedval = 5073)
incep_exp1_res100.to_csv('exp1_results_n100.csv', index=False)
rd_plt.plot1_prop_res(modelname = incep_name, modelcode = incep_code, full_results = incep_exp1_res100, n = 100)

'''
## EXPERIMENT 2 - SINGLE OBS ##
incep_exp2 = exp.SingleObs(inception_transformer)
incep_exp2_ref, incep_exp2_obs, incep_exp2_ref_PCA, incep_exp2_obs_PCA  = incep_exp2.run_exp_distance(ref_obs = 250, dist_size = 1000, new_obs = 100)


np.save('incep_exp2_ref.npy', incep_exp2_ref)
np.save('incep_exp2_obs.npy', incep_exp2_obs)
np.save('incep_exp2_ref_PCA.npy', incep_exp2_ref_PCA)
np.save('incep_exp2_obs_PCA.npy', incep_exp2_obs_PCA)
    
incep_exp2_class = incep_exp2.run_exp_classif(incep_exp2_ref, incep_exp2_obs)
incep_exp2_class_PCA = incep_exp2.run_exp_classif(incep_exp2_ref_PCA, incep_exp2_obs_PCA)
#incep_exp2_class_UMAP = incep_exp2.run_exp_classif(incep_exp2_ref_UMAP, incep_exp2_obs_UMAP)

incep_exp2_conf = utls.get_basic_confusion_vals(incep_exp2_class)
incep_exp2_conf_long = utls.convert_confusion_to_long(incep_exp2_conf)
incep_exp2_conf_PCA = utls.get_basic_confusion_vals(incep_exp2_class_PCA)
incep_exp2_conf_long_PCA = utls.convert_confusion_to_long(incep_exp2_conf_PCA)
#incep_exp2_conf_UMAP = utls.get_basic_confusion_vals(incep_exp2_class_UMAP)
#incep_exp2_conf_long_UMAP = utls.convert_confusion_to_long(incep_exp2_conf_UMAP)

incep_exp2_class.to_csv('exp2_class.csv', index = False)
incep_exp2_conf.to_csv('exp2_conf.csv', index = False)
incep_exp2_class_PCA.to_csv('exp2_class_PCA.csv', index = False)
incep_exp2_conf_PCA.to_csv('exp2_conf_PCA.csv', index = False)
#incep_exp2_class_UMAP.to_csv('exp2_class_UMAP.csv', index = False)
#incep_exp2_conf_UMAP.to_csv('exp2_conf_UMAP.csv', index = False)

rd_plt.plot2_classification(incep_name, incep_code, incep_exp2_conf_long, dim_red = None)
rd_plt.plot2_classification(incep_name, incep_code, incep_exp2_conf_long_PCA, dim_red = "PCA")
#rd_plt.plot2_classification(incep_name, incep_code, incep_exp2_conf_long_UMAP, dim_red = "UMAP")

incep_exp2_conf_adv = utls.get_adv_confusion_vals(incep_exp2_conf)
incep_exp2_conf_adv_PCA = utls.get_adv_confusion_vals(incep_exp2_conf_PCA)
#incep_exp2_conf_adv_UMAP = utls.get_adv_confusion_vals(incep_exp2_conf_UMAP)

rd_plt.plot2_roc_pr(incep_name, incep_code, incep_exp2_conf_adv, dim_red = None)
rd_plt.plot2_roc_pr(incep_name, incep_code, incep_exp2_conf_adv_PCA, dim_red = "PCA")
#rd_plt.plot2_roc_pr(incep_name, incep_code, incep_exp2_conf_adv_UMAP, dim_red = "UMAP")
'''
