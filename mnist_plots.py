# -*- coding: utf-8 -*-
"""
This script includes functions necessary to create the results plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def make_gan_examples(ganmodel, modelname):

    z = np.random.normal(0, 1, (8,100))
    pred = ganmodel.generator.predict(z)
    image = (pred+1) * 127.5

    fig, ax = plt.subplots(2,2)
    for i, ax in enumerate(ax.flatten()):
        plot = np.reshape(image[i+2], (28, 28))
        ax.imshow(plot, cmap='gray_r')
    plt.savefig(modelname +  "_gen_examples.png")
    plt.clf()

    print("GAN generator example plot created and saved!")

def make_results_plots(modelname, modelcode, wass_results, kmmd_results, frch_results, dim_red, pixel_space = False):
    if dim_red == None:
        dim_red = ""
    
    ax1 = sns.violinplot(data = wass_results[wass_results['space'] == 'feature'], x  = 'dataset', y = 'value', split = True)
    ax1 = ax1.set_title(modelname + ": Wasserstein distances in feature space")
    plt.savefig(modelcode +  "_violin_wass_fts_" + dim_red + ".png")
    plt.clf()
    
    if pixel_space == True:
        ax2 = sns.violinplot(data = wass_results[wass_results['space'] == 'pixel'], x  = 'dataset', y = 'value', split = True)
        ax2 = ax2.set_title(modelname + ": Wasserstein distances in pixel space")
        plt.savefig(modelcode +  "_violin_wass_pxl_" + dim_red + ".png")
        plt.clf()

    ax3 = sns.violinplot(data = wass_results[wass_results['space'] == 'feature'], x  = 'dataset', y = 'value', split = True)
    ax3 = ax3.set_title(modelname + ": kMMD distances in feature space")
    plt.savefig(modelcode +  "_violin_kmmd_fts_" + dim_red + ".png")
    plt.clf()

    if pixel_space == True:
        ax4 = sns.violinplot(data = wass_results[wass_results['space'] == 'pixel'], x  = 'dataset', y = 'value', split = True)
        ax4 = ax4.set_title(modelname + ": kMMD distances in pixel space")
        plt.savefig(modelcode +  "_violin_kmmd_pxl_" + dim_red + ".png")
        plt.clf()

    ax5 = sns.violinplot(data = wass_results[wass_results['space'] == 'feature'], x  = 'dataset', y = 'value', split = True)
    ax5 = ax5.set_title(modelname + ": Frechet distances in feature space")
    plt.savefig(modelcode +  "_violin_frch_fts_" + dim_red + ".png")
    plt.clf()

    if pixel_space == True:
        ax6 = sns.violinplot(data = wass_results[wass_results['space'] == 'pixel'], x  = 'dataset', y = 'value', split = True)
        ax6 = ax6.set_title(modelname + ": Frechet distances in pixel space")
        plt.savefig(modelcode +  "_violin_frch_pxl_" + dim_red + ".png")
        plt.clf()

    ax1 = sns.boxplot(data = wass_results[wass_results['space'] == 'feature'], x  = 'dataset', y = 'value')
    ax1 = ax1.set_title(modelname + ": Wasserstein distances in feature space")
    plt.savefig(modelcode +  "_box_wass_fts_" + dim_red + ".png")
    plt.clf()

    if pixel_space == True:
        ax2 = sns.boxplot(data = wass_results[wass_results['space'] == 'pixel'], x  = 'dataset', y = 'value')
        ax2 = ax2.set_title(modelname + ": Wasserstein distances in pixel space")
        plt.savefig(modelcode +  "_box_wass_pxl_" + dim_red + ".png")
        plt.clf()

    ax3 = sns.boxplot(data = wass_results[wass_results['space'] == 'feature'], x  = 'dataset', y = 'value')
    ax3 = ax3.set_title(modelname + ": kMMD distances in feature space")
    plt.savefig(modelcode +  "_box_kmmd_fts_" + dim_red + ".png")
    plt.clf()

    if pixel_space == True:
        ax4 = sns.boxplot(data = wass_results[wass_results['space'] == 'pixel'], x  = 'dataset', y = 'value')
        ax4 = ax4.set_title(modelname + ": kMMD distances in pixel space")
        plt.savefig(modelcode +  "_box_kmmd_pxl_" + dim_red + ".png")
        plt.clf()

    ax5 = sns.boxplot(data = wass_results[wass_results['space'] == 'feature'], x  = 'dataset', y = 'value')
    ax5 = ax5.set_title(modelname + ": Frechet distances in feature space")
    plt.savefig(modelcode +  "_box_frch_fts_" + dim_red + ".png")
    plt.clf()

    if pixel_space == True:
        ax6 = sns.boxplot(data = wass_results[wass_results['space'] == 'pixel'], x  = 'dataset', y = 'value')
        ax6 = ax6.set_title(modelname + ": Frechet distances in pixel space")
        plt.savefig(modelcode +  "_box_frch_pxl_" + dim_red + ".png")
        plt.clf()

    print("Results plots created and saved!")



def make_loss_plots(modelname, modelcode, losses, epochs):

    len_x = len(losses)+1

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
