#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised Geodesic SNE and t-SNE implementation

"""

import sys
import time
import warnings
import numpy as np
import scipy as sp
import networkx as nx
import sklearn.neighbors as sknn
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from sklearn import preprocessing
from numpy import log
from numpy import trace
from numpy import dot
from numpy import sqrt
from numpy.linalg import det
from numpy.linalg import inv
from numpy.linalg import norm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier


# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

"""Compute matrix containing geodesic distances

    # Arguments:
        X: matrix of size NxD
    # Returns:
        NxN matrix D, with entry D_ij = negative squared
        euclidean distance between rows X_i and X_j
    """
def neg_geodesic_dists(X, y):
    # Number of samples
    n = X.shape[0]
    m = X.shape[1]
    k = round(np.sqrt(n))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    W = knnGraph.toarray()
    G = nx.from_numpy_array(W)          # KNN graph
    # Geodesic distances    
    D = nx.floyd_warshall_numpy(G)
    # To remove inf's and nan's (when KNN graph is not connected)
    maximo = np.nanmax(D[D != np.inf])   
    D[np.isnan(D)] = 0    
    D[np.isinf(D)] = maximo
    # Supervised
    W2 = D.copy()
    for i in range(n):
        for j in range(n):
            euclidean = norm(X[i, :] - X[j, :])**2 
            if W[i, j] > 0:
                if y[i] != y[j]:
                    W2[i, j] = (D[i, j] + euclidean)
    return -W2


"""Take softmax of each row of matrix X."""
def softmax(X, diag_zero=True):
    # Subtract max for numerical stability
    e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
    # Original SNE computation
    #e_x = np.exp(X)
    # We usually want diagonal probailities to be 0.
    if diag_zero:
        np.fill_diagonal(e_x, 0.)
    # Add a tiny constant for stability of log we take later
    e_x = e_x + 1e-8  # numerical stability

    return e_x / e_x.sum(axis=1).reshape([-1, 1])

"""Convert a distances matrix to a matrix of probabilities."""
def calc_prob_matrix(distances, sigmas=None):
    if sigmas is not None:
        two_sig_sq = 2.0 * np.square(sigmas.reshape((-1, 1)))
        return softmax(distances / two_sig_sq)
    else:
        return softmax(distances)

"""Perform a binary search over input values to eval_fn.
    
    # Arguments
        eval_fn: Function that we are optimising over.
        target: Target value we want the function to output.
        tol: Float, once our guess is this close to target, stop.
        max_iter: Integer, maximum num. iterations to search for.
        lower: Float, lower bound of search range.
        upper: Float, upper bound of search range.
    # Returns:
        Float, best input value to function found during search.
    """
def binary_search(eval_fn, target, tol=1e-10, max_iter=10000, lower=1e-20, upper=1000.):
    for i in range(max_iter):
        guess = (lower + upper) / 2.0
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess

"""Calculate the perplexity of each row  of a matrix of probabilities."""
def calc_perplexity(prob_matrix):
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2 ** entropy
    
    return perplexity

"""Wrapper function for quick calculation of perplexity over a distance matrix."""
def perplexity(distances, sigmas):
    return calc_perplexity(calc_prob_matrix(distances, sigmas))

"""For each row of distances matrix, find sigma that results in target perplexity for that role."""
def find_optimal_sigmas(distances, target_perplexity):
    sigmas = [] 
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: perplexity(distances[i:i+1, :], np.array(sigma))
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)

    return np.array(sigmas)

"""SNE: Given low-dimensional representations Y, compute matrix of joint probabilities with entries q_ij."""
def q_joint(Y, y):
    # Get the distances from every point to every other
    distances = neg_geodesic_dists(Y, y)
    # Take the elementwise exponent
    exp_distances = np.exp(distances)
    # Fill diagonal with zeroes so q_ii = 0
    np.fill_diagonal(exp_distances, 0.)
    
    # Divide by the sum of the entire exponentiated matrix
    return exp_distances / np.sum(exp_distances), None

"""t-SNE: Given low-dimensional representations Y, compute matrix of joint probabilities with entries q_ij."""
def q_tsne(Y, y):
    distances = neg_geodesic_dists(Y, y)
    inv_distances = np.power(1. - distances, -1)
    np.fill_diagonal(inv_distances, 0.)
    
    return inv_distances / np.sum(inv_distances), inv_distances

"""Given conditional probabilities matrix P, return approximation of joint distribution probabilities."""
def p_conditional_to_joint(P):
    return (P + P.T) / (2. * P.shape[0])

"""Given a data matrix X, gives joint probabilities matrix.

    # Arguments
        X: Input data matrix.
    # Returns:
        P: Matrix with entries p_ij = joint probabilities.
"""
def p_joint(X, y, target_perplexity):
    # Get the negative euclidian distances matrix for our data
    distances = neg_geodesic_dists(X, y)
    # Find optimal sigma for each row of this distances matrix
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix(distances, sigmas)
    # Go from conditional to joint probabilities matrix
    P = p_conditional_to_joint(p_conditional)

    return P

"""SNE: Estimate the gradient of the cost with respect to Y"""
def symmetric_sne_grad(P, Q, Y, _):
    pq_diff = P - Q  # NxN matrix
    pq_expanded = np.expand_dims(pq_diff, 2)  #NxNx1
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  #NxNx2
    grad = 4. * (pq_expanded * y_diffs).sum(1)  #Nx2
    
    return grad

"""t-SNE: Estimate the gradient of t-SNE cost with respect to Y."""
def tsne_grad(P, Q, Y, inv_distances):
    pq_diff = P - Q
    pq_expanded = np.expand_dims(pq_diff, 2)
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
    # Expand our inv_distances matrix so can multiply by y_diffs
    distances_expanded = np.expand_dims(inv_distances, 2)
    # Multiply this by inverse distances matrix
    y_diffs_wt = y_diffs * distances_expanded
    # Multiply then sum over j's
    grad = 4. * (pq_expanded * y_diffs_wt).sum(1)
    
    return grad


""" Plot a 2D matrix with corresponding class labels: each class diff colour """
def categorical_scatter_2d(X2D, class_idxs, ms=3, ax=None, alpha=0.1, legend=True, figsize=None, show=False, savename=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    #ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    classes = list(np.unique(class_idxs))
    markers = 'os' * len(classes)
    colors = plt.cm.rainbow(np.linspace(0,1,len(classes)))

    for i, cls in enumerate(classes):
        mark = markers[i]
        ax.plot(X2D[class_idxs==cls, 0], X2D[class_idxs==cls, 1], marker=mark, linestyle='', ms=ms, label=str(cls), alpha=alpha, color=colors[i], markeredgecolor='black', markeredgewidth=0.4)
    if legend:
        ax.legend()
        
    if savename is not None:
        plt.tight_layout()
        plt.savefig(savename)
    
    if show:
        plt.show()
    
    return ax

"""Estimates a SNE model.

    # Arguments
        X: Input data matrix.
        y: Class labels for that matrix.
        P: Matrix of joint probabilities.
        rng: np.random.RandomState().
        num_iters: Iterations to train for.
        q_fn: Function that takes Y and gives Q prob matrix.
        plot: How many times to plot during training.
    # Returns:
        Y: Matrix, low-dimensional representation of X.
    """
def estimate_sne(X, y, P, rng, num_iters, q_fn, grad_fn, learning_rate, momentum, plot):

    # Initialise our 2D representation
    Y = rng.normal(0., 0.0001, [X.shape[0], 2])

    # Initialise past values (used for momentum)
    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

    # Start gradient descent loop
    # print('Iterations')
    for i in range(num_iters):
        print('Iteration ', i)

        # Get Q and distances (distances only used for t-SNE)
        Q, distances = q_fn(Y, y)
        # Estimate gradients with respect to Y
        grads = grad_fn(P, Q, Y, distances)

        # Update Y
        Y = Y - learning_rate * grads
        if momentum:  # Add momentum
            Y += momentum * (Y_m1 - Y_m2)
            # Update previous Y's for momentum
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()

    # Plot sometimes
    # if plot and i % (num_iters / plot) == 0:
    #     categorical_scatter_2d(Y, y, alpha=1.0, ms=6, show=True, figsize=(9, 6))
    categorical_scatter_2d(Y, y, alpha=1.0, ms=6, show=True, figsize=(9, 6))

    return Y


'''
 Computes the Silhouette coefficient and the supervised classification
 accuracies for several classifiers: KNN, SVM, NB, DT, QDA, MPL, GPC and RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 method: string to identify the DR method (PCA, NP-PCAKL, KPCA, ISOMAP, LLE, LAP, ...)
'''
def Classification(dados, target, mode='holdout'):
    
    lista = []
    lista_k = []

    # 8 different classifiers
    neigh = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(gamma='auto')
    nb = GaussianNB()
    dt = DecisionTreeClassifier(random_state=42)
    qda = QuadraticDiscriminantAnalysis()
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=1000)
    gpc = GaussianProcessClassifier()
    rfc = RandomForestClassifier()

    if mode == 'holdout':
        # 50% for training and 40% for testing
        X_train, X_test, y_train, y_test = train_test_split(dados.real, target, train_size=0.5, random_state=42)

        # KNN
        neigh.fit(X_train, y_train) 
        acc = neigh.score(X_test, y_test)
        labels_knn = neigh.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_knn, y_test)
        lista.append(acc)
        lista_k.append(kappa)

        # SMV
        svm.fit(X_train, y_train) 
        acc = svm.score(X_test, y_test)
        labels_svm = svm.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_svm, y_test)
        lista.append(acc)
        lista_k.append(kappa)
        

        # Naive Bayes
        nb.fit(X_train, y_train)
        acc = nb.score(X_test, y_test)
        labels_nb = nb.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_nb, y_test)
        lista.append(acc)
        lista_k.append(kappa)

        # Decision Tree
        dt.fit(X_train, y_train)
        acc = dt.score(X_test, y_test)
        labels_dt = dt.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_dt, y_test)
        lista.append(acc)
        lista_k.append(kappa)


        # Quadratic Discriminant 
        qda.fit(X_train, y_train)
        acc = qda.score(X_test, y_test)
        labels_qda = qda.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_qda, y_test)
        lista.append(acc)
        lista_k.append(kappa)

        # MPL classifier
        mpl.fit(X_train, y_train)
        acc = mpl.score(X_test, y_test)
        labels_mpl = mpl.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_mpl, y_test)
        lista.append(acc)
        lista_k.append(kappa)

        # Gaussian Process
        gpc.fit(X_train, y_train)
        acc = gpc.score(X_test, y_test)
        labels_gpc = gpc.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_gpc, y_test)
        lista.append(acc)
        lista_k.append(kappa)
        
        # Random Forest Classifier
        rfc.fit(X_train, y_train)
        acc = rfc.score(X_test, y_test)
        labels_rfc = rfc.predict(X_test)
        kappa = metrics.cohen_kappa_score(labels_rfc, y_test)
        lista.append(acc)
        lista_k.append(kappa)

        # Computes the average accuracy
        kap = max(lista_k)
        acc = max(lista)
        
        print()
        print('Maximum accuracy: ', acc)
        print('Maximum Kappa: ', kap)
        print()

        return [acc, kap]




if __name__ == '__main__':
    
    # Set global parameters
    PERPLEXITY = 20
    SEED = 1                    # Random seed
    MOMENTUM = 0.9
    LEARNING_RATE = 30
    NUM_ITERS = 50             # Num iterations to train for
    TSNE = False                # If False, Symmetric SNE
    NUM_PLOTS = 5               # Num. times to plot in training

    # numpy RandomState for reproducibility
    rng = np.random.RandomState(SEED)

    # Load data
    dados = skdata.load_iris()
    #dados = skdata.load_digits()
    #dados = skdata.fetch_openml(name='prnn_crabs', version=1) 
    #dados = skdata.fetch_openml(name='balance-scale', version=1)
    #dados = skdata.fetch_openml(name='parity5', version=1) 
    #dados = skdata.fetch_openml(name='hayes-roth', version=2)  
    #dados = skdata.fetch_openml(name='rabe_131', version=2)                        
    #dados = skdata.fetch_openml(name='servo', version=1)                       
    #dados = skdata.fetch_openml(name='monks-problems-1', version=1)            
    #dados = skdata.fetch_openml(name='bolts', version=2)                        
    #dados = skdata.fetch_openml(name='fri_c2_100_10', version=2)                   
    #dados = skdata.fetch_openml(name='threeOf9', version=1)                     
    #dados = skdata.fetch_openml(name='fri_c3_100_5', version=2)             
    #dados = skdata.fetch_openml(name='baskball', version=2)                     
    #dados = skdata.fetch_openml(name='newton_hema', version=2)                 
    #dados = skdata.fetch_openml(name='strikes', version=2)                      
    #dados = skdata.fetch_openml(name='datatrieve', version=1)        
    #dados = skdata.fetch_openml(name='diggle_table_a2', version=2) 
    #dados = skdata.fetch_openml(name='fl2000', version=2) 
    #dados = skdata.fetch_openml(name='triazines', version=2) 
    #dados = skdata.fetch_openml(name='veteran', version=2) 
    #dados = skdata.fetch_openml(name='diabetes', version=1) 
    #dados = skdata.fetch_openml(name='car', version=3)                          
    #dados = skdata.fetch_openml(name='prnn_fglass', version=2) 
    #dados = skdata.fetch_openml(name='analcatdata_creditscore', version=1) 
    #dados = skdata.fetch_openml(name='pwLinear', version=2)
    #dados = skdata.load_breast_cancer()                                        
    #dados = skdata.load_wine()                                                 
    #dados = skdata.fetch_openml(name='backache', version=1)                    
    #dados = skdata.fetch_openml(name='heart-statlog', version=1)
    
    X = dados['data']
    y = dados['target']

    # Optional: Reduce datasets (if n is too large)
    #X, lixo, y, garbage = train_test_split(X, y, train_size=0.25, random_state=42)

    # Treat catregorical features (required for some OpenML datasets)
    if not isinstance(X, np.ndarray):
        cat_cols = X.select_dtypes(['category']).columns
        X[cat_cols] = X[cat_cols].apply(lambda x: x.cat.codes)
        # Convert to numpy array
        X = X.to_numpy()
        y = y.to_numpy()

    n = X.shape[0]
    m = X.shape[1]
    c = len(np.unique(y))

    print('N = ', n)
    print('M = ', m)
    print('C = %d' %c)
    print()

    # Normalize data
    X = preprocessing.scale(X)

    # Obtain matrix of joint probabilities p_ij
    P = p_joint(X, y, PERPLEXITY)

    # Fit SNE or t-SNE
    Y = estimate_sne(X, y, P, rng, num_iters=NUM_ITERS, q_fn=q_tsne if TSNE else q_joint, grad_fn=tsne_grad if TSNE else symmetric_sne_grad, learning_rate=LEARNING_RATE, momentum=MOMENTUM, plot=NUM_PLOTS)

    # Classification
    medidas = Classification(Y, y)