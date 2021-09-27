# -*- coding: utf-8 -*-
"""
This script includes functions to generate all the results plots.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

#os.chdir('C:\\Users\\makos\\Documents\\Degree_MSc\\Year_3\\Thesis_cont\\code')

def make_gan_examples(ganmodel, modelname):

    z = np.random.normal(0, 1, (8,100))
    pred = ganmodel.generator.predict(z)
    image = (pred+1) * 127.5
    image = np.rint(image)
    fig, ax = plt.subplots(2,2)
    for i, ax in enumerate(ax.flatten()):
        plot = np.reshape(image[i+2], (32, 32, 3))
        ax.imshow(plot.astype('uint8'))
    plt.savefig(modelname +  "_gen_examples.png")
    plt.clf()

    print("GAN generator example plot created and saved!")

def plot1_prop_res(modelname, modelcode, full_results, n):
    """
    Generates plots of the distribution of distances according to proportion of F observations.
    1 PLOT PER Distance Measure
    """
    dims = ['none', 'PCA', 'UMAP']
    
    for dim in dims:
        if dim == 'none':
            dim_code = ""
        else:
            dim_code = "_" + dim
        if dim == 'none':
            dim_name = ""
        else:
            dim_name = ", (" + dim + ")"
    
        results = full_results[full_results['dim reduction'] == dim]
        plt.clf()    
        ax1 = sns.violinplot(data = results[results['measure'] == 'wasserstein'], x  = 'proportion', y = 'value', split = True)
        ax1 = ax1.set_title(modelname + ": Wasserstein distance (by proportion of true obs)" + dim_name)
        plt.savefig(modelcode +  "_wass_prop_violin" + dim_code + "_n" + str(n) + ".png")
        plt.clf()

        ax2 = sns.violinplot(data = results[results['measure'] == 'kmmd'], x  = 'proportion', y = 'value', split = True)
        ax2 = ax2.set_title(modelname + ": kMMD distance (by proportion of true obs)" + dim_name)
        plt.savefig(modelcode +  "_kmmd_prop_violin" + dim_code + "_n" + str(n) + ".png")
        plt.clf()

        ax3 = sns.violinplot(data = results[results['measure'] == 'frechet'], x  = 'proportion', y = 'value', split = True)
        ax3 = ax3.set_title(modelname + ": Frechet distance (by proportion of true obs)" + dim_name)
        plt.savefig(modelcode +  "_frch_prop_violin" + dim_code + "_n" + str(n) + ".png")
        plt.clf()

        ax1 = sns.boxplot(data = results[results['measure'] == 'wasserstein'], x  = 'proportion', y = 'value')
        ax1 = ax1.set_title(modelname + ": Wasserstein distance (by proportion of true obs)" + dim_name)
        plt.savefig(modelcode +  "_wass_prop_box" + dim_code + "_n" + str(n) + ".png")
        plt.clf()

        ax2 = sns.boxplot(data = results[results['measure'] == 'kmmd'], x  = 'proportion', y = 'value')
        ax2 = ax2.set_title(modelname + ": kMMD distance (by proportion of true obs)" + dim_name)
        plt.savefig(modelcode +  "_kmmd_prop_box" + dim_code + "_n" + str(n) + ".png")
        plt.clf()

        ax3 = sns.boxplot(data = results[results['measure'] == 'frechet'], x  = 'proportion', y = 'value')
        ax3 = ax3.set_title(modelname + ": Frechet distance (by proportion of true obs)" + dim_name)
        plt.savefig(modelcode +  "_frch_prop_box" + dim_code + "_n" + str(n) + ".png")
        plt.clf()
        
        print("plots for dimensionality reduction " + dim + " completed!")

    print("ALL plots created and saved!")

def make_loss_plots(modelname, modelcode, losses, epochs):

    len_x = len(losses)+1
    plt.clf()
    plt.plot(np.arange(1, len_x), losses[:, 0], color = 'r', label = "Discriminator loss")
    plt.plot(np.arange(1, len_x), losses[:, 1], color = 'g', label = "Generator loss")
    plt.legend(loc = "upper right")
    plt.title("Training " + modelname)
    plt.savefig(modelcode + "_losses.png")
    plt.clf()


    plt.plot(np.arange(1, len_x), losses[:, 2])
    plt.title("Training " + modelname + " discriminator accuracy")
    plt.savefig(modelcode + "_disc_acc.png")
    plt.clf()

    print("Loss plot created and saved!")

def plot2_classification(modelname, modelcode, confusion_res, dim_red):
    """
    Generates plots of TP< FP, TN, FN
    1 PLOT PER Distance Measure
    2 PERSPECTIVES - Split, Threshold
    """
    splits = np.unique(confusion_res['Split'])
    thresholds_all = np.unique(confusion_res['Threshold'])
    thresholds = thresholds_all[(thresholds_all >= 0.8) & (thresholds_all < 1)]*100
    
    if dim_red == None:
        red_code = ""
    
    elif dim_red == "PCA":
        red_code = "_PCA"

    elif dim_red == "UMAP":
        red_code = "_UMAP"
    
    for split_val in splits:
        threshold = confusion_res[confusion_res['Split'] == split_val]
        plt.clf()
        ax1 = sns.lineplot(data=threshold[threshold['Measure'] == 'Wasserstein'], y = "Result", x = "Threshold", hue = "Acc. Metric")
        ax1.set_title("Classification perf. of Wasserstein by threshold value")
        ax1.set_xticks(np.unique(threshold['Threshold']))
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        plt.savefig(modelcode + "_split" + str(split_val) +  "_wass_thresh_ALL" + red_code + ".png")
        plt.clf()

        ax2 = sns.lineplot(data=threshold[threshold['Measure'] == 'kMMD'], y = "Result", x = "Threshold", hue = "Acc. Metric")
        ax2.set_title("Classification perf. of kMMD by threshold value")
        ax2.set_xticks(np.unique(threshold['Threshold']))
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
        plt.savefig(modelcode + "_split" + str(split_val) +  "_kmmd_thresh_ALL" + red_code + ".png")
        plt.clf()

        ax3 = sns.lineplot(data=threshold[threshold['Measure'] == 'Frechet'], y = "Result", x = "Threshold", hue = "Acc. Metric")
        ax3.set_title("Classification perf. of Frechet by threshold value")
        ax3.set_xticks(np.unique(threshold['Threshold']))
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
        plt.savefig(modelcode + "_split" + str(split_val) +  "_frech_thresh_ALL" + red_code + ".png")
        plt.clf()


    for threshold_val in thresholds:
        split = confusion_res[confusion_res['Threshold'] == threshold_val/100]
        
        split["Split"].astype(float)
        plt.clf()        
        ax1 = sns.lineplot(data=splits[splits['Measure'] == 'Wasserstein'], y = "Result", x = "Split", hue = "Acc. Metric")
        ax1.set_title("Classification perf. of Wasserstein by split value")
        ax1.set_xticks(np.unique(split['Split']))
        plt.savefig(modelcode + "_thresh" + str(int(threshold_val*100)) + "_wass_split_ALL" + red_code + ".png")
        plt.clf()

        ax2 = sns.lineplot(data=splits[splits['Measure'] == 'kMMD'], y = "Result", x = "Split", hue = "Acc. Metric")
        ax2.set_title("Classification perf. of kMMD by split value")
        ax2.set_xticks(np.unique(split['Split']))
        plt.savefig(modelcode + "_thresh" + str(int(threshold_val*100)) +  "_kmmd_split_ALL" + red_code + ".png")
        plt.clf()

        ax3 = sns.lineplot(data=splits[splits['Measure'] == 'Frechet'], y = "Result", x = "Split", hue = "Acc. Metric")
        ax3.set_title("Classification perf. of Frechet by split value")
        ax3.set_xticks(np.unique(split['Split']))
        plt.savefig(modelcode + "_thresh" + str(int(threshold_val*100)) +  "_frech_split_ALL" + red_code + ".png")
        plt.clf()

    print("Plots created and saved!")


def plot2_roc_pr(modelname, modelcode, adv_confusion, dim_red):
    """
    Generates Precision-Recall curves and ROC curves
    1 PLOT PER Distance Measure
    """

    measures = adv_confusion["Measure"].unique()
    
    if dim_red == None:
        red_code = ""
    
    elif dim_red == "PCA":
        red_code = "_PCA"

    elif dim_red == "UMAP":
        red_code = "_UMAP"

    for measure in measures:
        plt.clf()
        selection = adv_confusion[(adv_confusion["Measure"] == measure)]
        ax1 = sns.lineplot(data = selection, y = "Recall", x = "Precision", drawstyle='steps-pre', hue = "Split")
        ax1.set_title("Precision-Recall curve for " + measure + ".")
        plt.savefig(modelcode + "_" + measure + "_PrecRec_curve" + red_code + ".png")
        plt.clf()

        ax2 = sns.lineplot(data = selection, y = "TP rate", x = "FP rate", drawstyle='steps-pre', hue = "Split")
        ax2.set_title("ROC curve for " + measure + ".")
        plt.savefig(modelcode + "_" + measure + "_ROC_curve" + red_code + ".png")
        plt.clf()

    print("Plots created and saved")

def plot2_ind_distances(long_results, modelcode, modelname, dim_name):
    splits = np.unique(long_results[:, 3])
    if dim_name is not None:
        dim_code = "_" + dim_name
        dim_name = " (" + dim_name + ")"
    else:
        dim_name = ""
        dim_code = ""
    for split in splits:
        red_split = long_results[long_results[:,3] == split, :]
            
        results = pd.DataFrame(red_split, columns = ['value', 'observations', 'measure', 'split'])
        results['value'] = results['value'].astype('float64')
        results['observations'] = results['observations'].astype('category')
        
        ax1 = sns.boxplot(data = results[results['measure'] == 'Wasserstein'], x  = 'observations', y = 'value')
        ax1 = ax1.set_title(modelname + ": Wasserstein distance" + dim_name)
        plt.savefig(modelcode +  "_wass_prop_box" + dim_code + "_split" + str(split.replace('.', '')) + ".png")
        plt.clf()

        ax1 = sns.boxplot(data = results[results['measure'] == 'Kernel MMD'], x  = 'observations', y = 'value')
        ax1 = ax1.set_title(modelname + ": Kernel MMD distance" + dim_name)
        plt.savefig(modelcode +  "_kmmd_prop_box" + dim_code + "_split" + str(split.replace('.', '')) + ".png")
        plt.clf()

        ax1 = sns.boxplot(data = results[results['measure'] == 'Frechet'], x  = 'observations', y = 'value')
        ax1 = ax1.set_title(modelname + ": Frechet distance" + dim_name)
        plt.savefig(modelcode +  "_frch_prop_box" + dim_code + "_split" + str(split.replace('.', '')) + ".png")
        plt.clf()
    print("Plots created and saved!")

