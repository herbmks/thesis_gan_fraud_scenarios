# -*- coding: utf-8 -*-
"""

This script includes functions that are to be used in order to calculate the evaluation measures.

"""
import tensorflow as tf
import ot
import numpy as np
import math
from scipy import linalg


# Matrix Square Root funcitons

# Matrix square root using the Newton-Schulz method
def sqrt_newton_schulz(A, iterations=15, dtype='float64'):
    dim = A.shape[0]
    normA = tf.norm(A)
    Y = tf.divide(A, normA)
    I = tf.eye(dim, dtype=dtype)
    Z = tf.eye(dim, dtype=dtype)
    for i in range(iterations):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    sqrtA = Y * tf.sqrt(normA)
    return sqrtA


def matrix_sqrt_eigen(mat):
    eig_val, eig_vec = tf.linalg.eigh(mat)
    diagonal = tf.linalg.diag(tf.pow(eig_val, 0.5))
    mat_sqrt = tf.matmul(diagonal, tf.transpose(eig_vec))
    mat_sqrt = tf.matmul(eig_vec, mat_sqrt)
    return mat_sqrt



# Distance Metrics

def distance(X, Y, sqrt):
    # squared norms of each row in A and B
    nX = tf.math.reduce_sum(tf.square(X), 1)
    nY = tf.math.reduce_sum(tf.square(Y), 1)
    # na as a row and nb as a column vectors
    nX = tf.reshape(nX, [-1, 1])
    nY = tf.reshape(nY, [1, -1])
    # return pairwise euclidean difference matrix
    if sqrt == True:
        s1 = 2*tf.linalg.matmul(X, Y, False, True)
        #print("nX: ", tf.shape(nX), "; s1: ", tf.shape(s1), "; nY: ", tf.shape(nY))
        M = tf.math.sqrt(tf.math.maximum(nX - s1 + nY, 0.0))
    else:
        s1 = 2*tf.linalg.matmul(X, Y, False, True)
        #print("nX: ", tf.shape(nX), "; s1: ", tf.shape(s1), "; nY: ", tf.shape(nY))
        M = tf.math.maximum(nX - s1 + nY, 0.0)

    return M

def wasserstein(M, sqrt):
    if sqrt:
        M = tf.math.sqrt(M)
    emd = ot.emd2([], [], np.asarray(M))

    return emd

def kmmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.numpy().mean()
    Mxx = tf.math.exp(-Mxx/(scale*2*sigma*sigma))
    Mxy = tf.math.exp(-Mxy/(scale*2*sigma*sigma))
    Myy = tf.math.exp(-Myy/(scale*2*sigma*sigma))
    a = Mxx.numpy().mean() + Myy.numpy().mean() - 2*Mxy.numpy().mean()
    ans = math.sqrt(max(a, 0))

    return ans

def frechet(X, Y, mx_sqrt = "Shulz"):
    mu1, sigma1 = X.mean(axis=0), np.cov(X, rowvar=False)
    mu2, sigma2 = Y.mean(axis=0), np.cov(Y, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    sig_prod = sigma1.dot(sigma2)
    #print("sigprod obtained")
    if mx_sqrt == "Shulz":
        covmean = sqrt_newton_schulz(sig_prod)
    elif mx_sqrt == "Eigen":
        covmean = matrix_sqrt_eigen(sig_prod)
    else:
        covmean = linalg.sqrtm(sig_prod)
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        #print('is complex')
        covmean = covmean.real
    # calculate score
    frechet = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return frechet

def get_all_scores(real, fake, sigma = 1, sqrt = True):
    Mxx = distance(real, real, False)
    Mxy = distance(real, fake, False)
    Myy = distance(fake, fake, False)

    s_emd = wasserstein(Mxy, sqrt)
    s_kmmd = kmmd(Mxx, Mxy, Myy, sigma)
    s_frechet = frechet(real, fake)

    return s_emd, s_kmmd, s_frechet

def get_wasserstein(real, fake, sqrt = True):
    Mxy = distance(real, fake, False)

    s_emd = wasserstein(Mxy, sqrt)

    return s_emd

def get_kmmd(real, fake, sigma = 1):
    Mxx = distance(real, real, False)
    Mxy = distance(real, fake, False)
    Myy = distance(fake, fake, False)

    s_kmmd = kmmd(Mxx, Mxy, Myy, sigma)

    return s_kmmd
