# Overall Information

## What and Where?

M.Sc. Statistics and Data Science
Katholieke Universiteit Leuven - Leuven, Belgium.

## Title?

Statistical measures for evaluating generative models with an application to fraud detection.

## Author?

Blazej Herbert Makosa.

## Defence information?

Completed and passed September 2021.

## Aims of the thesis?

In this thesis we investigate two main topics. The first topic is the use of generative adversarial network model evaluation measures for fraud, or outlier detection. These evaluation measures determine the distance between samples of observations, usually a sample from the target population and a sample of generated observation. The second topic is the possibility, or viability, of using the generative adversarial network’s discriminator as a feature space, in which the aforementioned distance measures could be calculated in.

The setting for the work involves the idea of applying the two topics to the problem of unsupervised fraud detection. More specifically, the environment in which the behaviours are explored supposes that there are no examples of fraudulent observations available during training. This means that the problem is an outlier detection problem. Where the distance measures are used to determine a distance for new observations, and this distance is then used to determine whether the new observations are outliers or not. If they are, they are considered fraudulent samples or observations.

GAN models are trained on the available, non-fraudulent, data in order to obtain a feature space from the GAN discriminator. The distances between the known non-fraudulent observations and unknown fraudulent and non-fraudulent observations are calculated in the feature space created by the GAN discriminator network. There are three main experiments. Each focusing on different factors that may affect the behaviours of the distances that are calculated. These factors include: the type of GAN model used, the inclusion of dimensionality reduction techniques, and the proportion of fraudulent observations in the sample.


# Repo and Code Information

The code in this repository contains three main scripts in which the models are trained, the expereiments are performed, and the results are analysed, and the results plots are generated. These main scripts are: simple_main.py, mnist_main.py, cifar_main.py. The reaiming scripts include the necessary classes and functions in order for the main scripts to function as intended. The final thesis submission is included as the thesis.pdf file.