
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random as rnd
from scipy.stats import multivariate_normal

rnd.seed(2021)  # to reproduce the reported results
dimension: int = 2  # number of covariates for an instance
K = 3  # number of Gaussians


# to define each multivariate Gaussian distribution
@dataclass
class multigaussdistribution:
    mean: np.array(np.zeros([1, dimension], dtype=float))
    covariance: np.array(np.zeros([dimension, dimension], dtype=float))
    pi: float


# initialization of the means, covariances and phi for each distribution by random sampling, initial phi values are
# assumed to be equal it creates K many distributions
def initialization():
    distributionlist = []
    datasize = len(datasetmodified)
    sizeofsample = int(np.round(datasize / K, 0))  # to generate equal sized random samples from data
    indices = []
    for ind in range(datasize):
        indices.append(ind)
    samples = []
    for s in range(K):
        list1 = rnd.sample(indices, sizeofsample)
        samples.append(list1)

    for k in range(K):
        meantemp = []
        covtemp = np.random.normal(0, 1, size=(dimension, dimension))
        for d in range(dimension):
            meantemp.append(np.mean(datasetmodified[samples[k], d]))
            covtemp[d][d] = np.var(datasetmodified[samples[k], d])
        temp_dist = multigaussdistribution(mean=meantemp,
                                           covariance=covtemp, pi=1 / 3)
        distributionlist.append(temp_dist)
    return distributionlist


def expectationstep(data):
    for n in range(len(data)):
        denominator_n = 0
        for k in range(K):
            denominator_n += distributionList[k].pi * multivariate_normal.pdf(x=data[n, range(dimension)],
                                                                              mean=distributionList[k].mean,
                                                                              cov=distributionList[k].covariance)
        for k in range(K):
            data[n, dimension + k] = (distributionList[k].pi * multivariate_normal.pdf(x=data[n, range(dimension)],
                                                                                       mean=distributionList[k].mean,
                                                                                       cov=distributionList[
                                                                                           k].covariance)) / denominator_n
    return data


def maximizationstep(data):
    for k in range(K):
        N_k = 0
        for n in range(len(data)):
            N_k += data[n, dimension + k]

        mu_new = []
        for d in range(dimension):
            mu_new.append(0)

        # calculate new means
        for n in range(len(data)):
            mu_new += (1 / N_k) * data[n, dimension + k] * data[n, range(dimension)]

        # calculate new covariance
        covnew = np.array(np.zeros([dimension, dimension], dtype=float))
        for n in range(len(data)):
            diff = np.asmatrix((data[n, range(dimension)] - mu_new))
            covnew += (1 / N_k) * data[n, dimension + k] * (np.dot(diff.transpose(), diff))

        distributionList[k].mean = mu_new
        distributionList[k].covariance = covnew
        distributionList[k].pi = N_k / len(data)

    return distributionList


def convergencecheck(prevLogLLvalue, data):
    currentLogLikeliHoodValue = 0
    for n in range(len(data)):
        sum_n = 0
        for k in range(K):
            sum_n += (distributionList[k].pi * multivariate_normal.pdf(x=data[n, range(dimension)],
                                                                       mean=distributionList[k].mean,
                                                                       cov=distributionList[k].covariance))
        currentLogLikeliHoodValue += np.log(sum_n)

    difference = np.abs(prevLogLLvalue - currentLogLikeliHoodValue)
    if difference < 0.00001:  # must be less than 10 ^-5
        return True, currentLogLikeliHoodValue

    else:
        return False, currentLogLikeliHoodValue


# Overall Algorithm
max = -1000000000
dataset = np.load('dataset.npy')
for rep in range(10):
    rnd.seed(rep + 200)
    datasetmodified = np.hstack((dataset, np.zeros((dataset.shape[0], K), dtype=dataset.dtype)))
    distributionList = initialization()
    # add k many z variables for each instance

    prevLogLikelihoodValue = 0
    iter = 0
    outF = open("LLconvergence" + str(rep) + ".txt", "w")
    outF.close()
    while True:
        iter += 1
        datasetmodified = expectationstep(datasetmodified)
        distributionList = maximizationstep(datasetmodified)
        check, value = convergencecheck(prevLogLikelihoodValue, datasetmodified)
        outF = open("LLconvergence" + str(rep) + ".txt", "a")
        outF.write(str(value))
        outF.write("\n")
        outF.close()
        if check:
            break
        else:
            prevLogLikelihoodValue = value
    # reporting replication results
    datasetwithClusters = np.hstack(
        (datasetmodified, np.zeros((datasetmodified.shape[0], 1), dtype=datasetmodified.dtype)))  # cluster assignments
    # assigning instances to the clusters (distributions) from 1 to K clusters
    for n in range(len(datasetmodified)):
        maxProb = np.amax(datasetmodified[n, (dimension):(dimension + K)])
        index = np.where(datasetmodified[n, (dimension):(dimension + K)] == maxProb)
        datasetwithClusters[n, dimension + K] = int(index[0]) + 1

    plt.scatter(datasetwithClusters[:, 0], datasetwithClusters[:, 1],
                c=10 * (datasetwithClusters[:, dimension + K]))
    plt.savefig("clusterplot" + str(rep) + ".png")
    outF = open("Results" + str(rep) + ".txt", "w")
    for k in range(K):
        outF.write("Mean:" + "\t" + str(distributionList[k].mean) + "\t")
        for row in distributionList[k].covariance:
            outF.write("Covariance:" + "\n" + str(row) + "\t")
        outF.write("pi" + str(distributionList[k].pi))
        outF.write("\n")
    outF.close()
    if value > max:
        solution = datasetmodified
        max = value
        bestrep = rep
        bestdistributionFit = distributionList

# reporting best result among replications
datasetwithClusters = np.hstack(
    (solution, np.zeros((solution.shape[0], 1), dtype=solution.dtype)))  # cluster assignments
# assigning instances to the clusters (distributions) from 1 to K clusters
for n in range(len(datasetwithClusters)):
    maxProb = np.amax(solution[n, dimension:(dimension + K)])
    index = np.where(solution[n, dimension:(dimension + K)] == maxProb)
    datasetwithClusters[n, dimension + K] = int(index[0]) + 1

plt.scatter(datasetwithClusters[:, 0], datasetwithClusters[:, 1],
            c=10 * (datasetwithClusters[:, dimension + K]))
plt.show()
plt.savefig("clusterplotFinalBestrep" + str(bestrep) + ".png")
outF = open("Results.txt", "w")
for k in range(K):
    outF.write("Mean:" + "\t" + str(bestdistributionFit[k].mean) + "\t")
    for row in bestdistributionFit[k].covariance:
        outF.write("Covariance:" + "\n" + str(row) + "\t")
    outF.write("pi" + str(bestdistributionFit[k].pi))
    outF.write("\n")
outF.close()
