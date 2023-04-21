from operator import matmul
import numpy as np
import pandas as pd
import scipy.stats
import scipy.linalg as scilin
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import time
import random
from matplotlib.patches import Ellipse
from sklearn import mixture
import matplotlib.transforms as transforms
from functools import partial
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import torch.nn as nn
import torch.nn.functional as F

###################################################################################
# This is the associated code for the following paper
# "Fast Variational Estimation of Mutual Information for Implicit and Explicit
# Likelihood Models" by Caleb Dahlke, Sue Zheng, and Jason Pacheco.
# This runs the Gaussian Mixture Model experiment in section 8.1
###################################################################################


class FullyConnected(nn.Module):
    """
    Fully-connected neural network written as a child class of torch.nn.Module,
    used to compute the mutual information between two random variables.
    Code from S. Kleinegesse and M. U. Gutmann. "Bayesian experimental
    design for implicit models by mutual information neural
    estimation"
    https://github.com/stevenkleinegesse/minebed
    Attributes
    ----------
    self.fc_var1: torch.nn.Linear object
        Input layer for the first random variable.
    self.fc_var2: torch.nn.Linear object
        Input layer for the second random variable.
    self.layers: torch.nn.ModuleList object
        Object that contains all layers of the neural network.
    Methods
    -------
    forward:
        Forward pass through the fully-connected eural network.
    """

    def __init__(self, var1_dim, var2_dim, L=1, H=10):
        """
        Parameters
        ----------
        var1_dim: int
            Dimensions of the first random variable.
        var2_dim: int
            Dimensions of the second random variable.
        L: int
            Number of hidden layers of the neural network.
            (default is 1)
        H: int or np.ndarray
            Number of hidden units for each hidden layer. If 'H' is an int, all
            layers will have the same size. 'H' can also be an nd.ndarray,
            specifying the sizes of each hidden layer.
            (default is 10)
        """

        super(FullyConnected, self).__init__()

        # check for the correct dimensions
        if isinstance(H, (list, np.ndarray)):
            assert len(H) == L, "Incorrect dimensions of hidden units."
            H = list(map(int, list(H)))
        else:
            H = [int(H) for _ in range(L)]

        # Define layers over your two random variables
        self.fc_var1 = nn.Linear(var1_dim, H[0])
        self.fc_var2 = nn.Linear(var2_dim, H[0])

        # Define any further layers
        self.layers = nn.ModuleList()
        if L == 1:
            fc = nn.Linear(H[0], 1)
            self.layers.append(fc)
        elif L > 1:
            for idx in range(1, L):
                fc = nn.Linear(H[idx - 1], H[idx])
                self.layers.append(fc)
            fc = nn.Linear(H[-1], 1)
            self.layers.append(fc)
        else:
            raise ValueError('Incorrect value for number of layers.')

    def forward(self, var1, var2):
        """
        Forward pass through the neural network.
        Parameters
        ----------
        var1: torch.autograd.Variable
            First random variable.
        var2: torch.autograd.Variable
            Second random variable.
        """

        # Initial layer over random variables
        hidden = F.relu(self.fc_var1(var1) + self.fc_var2(var2))

        # All subsequent layers
        for idx in range(len(self.layers) - 1):
            hidden = F.relu(self.layers[idx](hidden))

        # Output layer
        output = self.layers[-1](hidden)

        return output

from tqdm import tqdm as tqdm

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


class MINE:
    """
    A class used to train a lower bound on the mutual information between two
    random variables.
    Code from S. Kleinegesse and M. U. Gutmann. "Bayesian experimental
    design for implicit models by mutual information neural
    estimation"
    https://github.com/stevenkleinegesse/minebed
    Attributes
    ----------
    model: torch.nn.Module (or child class) object
        A parametrised neural network that is trained to compute a lower bound
        on the mutual information between two random variables.
    optimizer: torch.optim object
        The optimiser used to learn the neural network parameters.
    scheduler: torch.optim.lr_scheduler object
        The learning rate scheduler used in conjunction with the optimizer.
    LB_type: str
        The type of mutual information lower bound that is maximised.
    X: np.ndarray
        Numpy array of the first random variable
    Y: np.ndarray
        Numpy array of the second random variable
    train_lb: np.ndarray
        Numpy array of the lower bound evaluations as a function of neural
        network parameter updates during training.
    Methods
    -------
    set_optimizer:
        Set a optimizer to train the neural network (recommended).
    set_scheduler:
        Set a scheduler to update the optimizer (recommended).
    evaluate_model:
        Evaluate the neural network for given two data points.
    evaluate_lower_bound:
        Evaluate the lower bound for two sets of data points.
    train:
        Train the neural network with the mutual information lower bound as
        the objective function to be maximised.
    """

    def __init__(
            self, model, data, LB_type='NWJ',
            lr=1e-3, schedule_step=1e8, schedule_gamma=1):
        """
        Parameters
        ----------
        model: torch.nn.Module (or child class) object
            A parametrised neural network that is trained to compute a lower
            bound on the mutual information between two random variables.
        data: tuple of np.ndarrays
            Tuple that contains the datasets of the two random variables.
        LB_type: str
            The type of mutual information lower bound that is maximised.
            (default is 'NWJ', also known as MINE-f)
        lr: float
            Learning rate of the Adam optimiser. May ignore if optimizer is
            specified later via the set_optimizer() method.
            (default is 1e-3)
        schedule_step: int
            Step size of the StepLR scheduler. May ignore if scheduler is
            specified later via the set_scheduler() method.
            (default is 1e8, sufficiently large to not be used by default)
        schedule_gamma: float
            Learning rate decay factor (gamma) of the StepLR scheduler. May
            ignore if scheduler is specified later via the set_scheduler()
            method. Should be between 0 and 1.
            (default is 1)
        """

        self.model = model
        self.X, self.Y = data
        self.LB_type = LB_type

        # default optimizer is Adam; may over-write
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        # default scheduler is StepLR; may over-write
        self.scheduler = StepLR(
            self.optimizer, step_size=schedule_step, gamma=schedule_gamma)

    def set_optimizer(self, optimizer):
        """
        Set a custom optimizer to be used during training.
        Parameters
        ----------
        optimizer: torch.optim object
            The custom optimizer object.
        """

        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        """
        Set a custom learning rate scheduler to be used during training.
        Parameters
        ----------
        scheduler: torch.optim.lr_scheduler object
            The custom optimizer object.
        """

        self.scheduler = scheduler

    def _ma(self, a, window=100):
        """Computes the moving average of array a, within a 'window'"""

        avg = [np.mean(a[i:i + window]) for i in range(0, len(a) - window)]
        return avg

    def _lower_bound(self, pj, pm):
        """Evaluates the lower bound with joint/marginal samples."""

        if self.LB_type == 'NWJ':
            # Compute the NWJ bound (also known as MINE-f)
            Z = torch.tensor(np.exp(1))
            lb = torch.mean(pj) - torch.mean(torch.exp(pm) / Z)
        else:
            raise NotImplementedError()

        return lb

    def evaluate_model(self, X, Y):
        """
        Evaluates the current model given two data points/sets 'X' and 'Y'.
        Parameters
        ----------
        X: np.ndarray of shape (:, dim(X))
            Numpy array of samples from the first random variable.
        Y: np.ndarray of shape (:, dim(Y))
            Numpy array of samples from the second random variable.
        """

        # Define PyTorch variables
        x = Variable(
            torch.from_numpy(X).type(torch.FloatTensor),
            requires_grad=True)
        y = Variable(
            torch.from_numpy(Y).type(torch.FloatTensor),
            requires_grad=True)

        # Get predictions from network
        predictions = self.model(x, y)

        return predictions

    def evaluate_lower_bound(self, X, Y):
        """
        Evaluates the lower bound using the current model and samples of the
        first and second random variable.
        Parameters
        ----------
        X: np.ndarray of shape (:, dim(X))
            Numpy array of samples from the first random variable.
        Y: np.ndarray of shape (:, dim(Y))
            Numpy array of samples from the second random variable.
        """

        # shuffle data
        Y_shuffle = np.random.permutation(Y)

        # Get predictions from network
        pred_joint = self.evaluate_model(X, Y)
        pred_marginal = self.evaluate_model(X, Y_shuffle)

        # Compute lower bound
        lb = self._lower_bound(pred_joint, pred_marginal)

        return lb

    def train(self, n_epoch, batch_size=None, bar=True):
        """
        Trains the neural network using samples of the first random variable
        and the second variable. The resulting objective function is stored
        in 'self.train_lb'.
        Parameters
        ----------
        n_epoch: int
            The number of training epochs.
        batch_size: int
            The batch size of data samples used during training.
            (default is None, in which case no batches are used)
        bar: boolean
            Displays a progress bar of the training procedure.
            (default is True)
        """

        # if no batch_size is given, set it to the size of the training set
        if batch_size is None:
            batch_size = len(self.X)

        # start the training procedure
        self.train_lb = []
        for epoch in tqdm(range(n_epoch), leave=True, disable=not bar):

            for b in range(int(len(self.X) / batch_size)):

                # sample batches randomly
                index = np.random.choice(
                    range(len(self.X)), size=batch_size, replace=False)
                x_sample = self.X[index]
                y_sample = self.Y[index]

                # Compute lower bound
                lb = self.evaluate_lower_bound(x_sample, y_sample)

                # maximise lower bound
                loss = - lb

                # save training score
                self.train_lb.append(lb.data.numpy())

                # parameter update steps
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            # scheduler step
            self.scheduler.step()

        self.train_lb = np.array(self.train_lb).reshape(-1)


##################################################
def GaussianMixtureParams(M,Dx,Dy):
    ###############################################################################
    # Outline: Randomly Generates Parameters for GMM
    #
    # Inputs:
    #       M - Number of components
    #       Dx - Number of dimensions for Latent Variable, X
    #       Dy - Number of dimensions for Observation Variable, Y
    #
    # Outputs:
    #       w - weights of components
    #       mu - means of components
    #       Sigma - Variance of components
    ###############################################################################
    D = Dx+Dy
    w = np.random.dirichlet(np.ones(M))
    mu = []
    sigma = []
    for d in range(M):
        # mu.append(np.random.uniform(-5,5,(D,1)))
        # A = np.random.rand(D, D)
        # B = np.dot(A, A.transpose())
        # sigma.append(B)
        mean = np.zeros((D,1))
        cov = 1*np.eye(D)+40*np.ones((D,D))
        mu.append(np.random.multivariate_normal(mean.flatten(),cov).reshape(D,1))
        B = 1*np.eye(D)+np.random.uniform(1,30)*np.ones((D,D))
        sigma.append(B)
    return w,mu,sigma

def SampleGMM(N,w,mu,sigma):
    ###############################################################################
    # Outline: Samples Points from a GMM
    #
    # Inputs:
    #       N - Number of points to sample
    #       w - weights of GMM components
    #       mu - means of GMM components
    #       Sigma - Variance of GMM components
    #
    # Outputs:
    #       samples - coordinates of sampled points
    ###############################################################################
    samples = np.zeros((N,len(mu[0])))
    for j in range(N):
        acc_pis = [np.sum(w[:i]) for i in range(1, len(w)+1)]
        r = np.random.uniform(0, 1)
        k = 0
        for i, threshold in enumerate(acc_pis):
            if r < threshold:
                k = i
                break
        x = np.random.multivariate_normal(mu[k].T.tolist()[0],sigma[k].tolist())
        samples[j,:] = x
    return samples

def MargEntGMM(sample,Dx,w,mu,Sigma):
    ###############################################################################
    # Outline: Numerically Calculates Marginal Entropy
    #
    # Inputs:
    #       samples - List of full sample set
    #       Dx - Dimension of Latent Variable, X
    #       w - weights of components
    #       mu - means of components
    #       Sigma - Variance of components
    #
    # Outputs:
    #       MargEnt - Marginal Entropy
    ###############################################################################
    M = len(w)
    x = sample[:,0:Dx]
    MargEntPart = np.zeros((M,len(sample[:,0])))
    for d in range(M):
        MargEntPart[d,:] = multivariate_normal.logpdf(x,mu[d][0:Dx].T.tolist()[0],Sigma[d][0:Dx,0:Dx])+np.log(w[d])
    MargEnt = -1*(1/len(sample[:,0]))*sum(logsumexp(MargEntPart,axis=0))
    return MargEnt

def CondEntGMM(sample,Dx,Dy,w,mu,Sigma):
    ###############################################################################
    # Outline: Numerically Calculates Marginal Entropy
    #
    # Inputs:
    #       samples - List of full sample set
    #       Dx - Dimension of Latent Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #       w - weights of components
    #       mu - means of components
    #       Sigma - Variance of components
    #
    # Outputs:
    #       CondEnt - Conditional Entropy
    ###############################################################################
    M = len(w)
    y = sample[:,Dx:(Dx+Dy)]
    JointEntPart = np.zeros((M,len(sample[:,0])))
    MargEntPart = np.zeros((M,len(sample[:,0])))
    for d in range(M):
        JointEntPart[d,:] = multivariate_normal.logpdf(sample,mu[d].T.tolist()[0],Sigma[d])+np.log(w[d])
        MargEntPart[d,:] = multivariate_normal.logpdf(y,mu[d][Dx:(Dx+Dy)].T.tolist()[0],Sigma[d][Dx:(Dx+Dy),Dx:(Dx+Dy)])+np.log(w[d])
    JointEnt = -1*sum(logsumexp(JointEntPart,axis=0))*(1/len(sample[:,0]))
    MargEnt = -1*sum(logsumexp(MargEntPart,axis=0))*(1/len(sample[:,0]))
    CondEnt = JointEnt-MargEnt
    return CondEnt

def pmoments(sample,Dx,Dy):
    ###############################################################################
    # Outline: Calculates the Moments of a GMM
    #
    # Inputs:
    #       samples - List of full sample set
    #       Dx - Dimension of Latent Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #
    # Outputs:
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPYY - Second moment of P w.r.t. Y
    #       EPXY - Second moment of P w.r.t. Cross Term XY
    ###############################################################################
    EPX = sum(sample[:,0:Dx])/len(sample[:,0])
    EPY = sum(sample[:,Dx:(Dx+Dy)])/len(sample[:,0])
    EPXX = np.matmul(sample[:,0:Dx].T,sample[:,0:Dx])/len(sample[:,0])
    EPXY = np.matmul(sample[:,0:Dx].T,sample[:,Dx:(Dx+Dy)])/len(sample[:,0])
    EPYY = np.matmul(sample[:,Dx:(Dx+Dy)].T,sample[:,Dx:(Dx+Dy)])/len(sample[:,0])
    return EPX, EPY, EPXX, EPXY, EPYY

def MargEntMomentMatch(Dx,EPX, EPXX):
    ###############################################################################
    # Outline: Calculates Implicit Likelihood Variational Marginal Entropy by 
    #          Moment Matching
    #
    # Inputs:
    #       Dx - Dimension of Latent Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPXX - Second moment of P w.r.t. X
    #
    # Outputs:
    #       VarMargEnt - Implicit Likelihood Variational Marginal Entropy
    ###############################################################################
    sigmaqx = EPXX-np.outer(EPX,EPX)
    VarMargEnt = .5*(Dx*np.log(2*np.pi)+np.log(np.linalg.det(sigmaqx))+Dx)
    return VarMargEnt

def CondEntMomentMatch(Dx,EPX, EPY, EPXX, EPXY, EPYY):
    ###############################################################################
    # Outline: Calculates Implicit Likelihood Variational Conditional Entropy by 
    #          Moment Matching
    #
    # Inputs:
    #       Dx - Dimension of Latent Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Cross Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #
    # Outputs:
    #       VarCondEnt - Implicit Likelihood Variational Conditional Entropy
    ###############################################################################
    sigmaqx = EPXX-np.outer(EPX,EPX)
    sigmaqy = EPYY-np.outer(EPY,EPY)
    sigmaqxy = EPXY-np.outer(EPX,EPY)
    condsigma = sigmaqx-np.matmul(sigmaqxy,np.matmul(np.linalg.inv(sigmaqy),sigmaqxy.T))
    VarCondEnt = .5*(Dx*np.log(2*np.pi)+np.log(np.linalg.det(condsigma))+Dx)
    return VarCondEnt

def MargEntGradientDescent(Dx,EPX,EPXX,Tol,FullOut=False):
    ###############################################################################
    # Outline: Calculates Implicit Likelihood Variational Marginal Entropy by 
    #          Gradient Descent
    #
    # Inputs:
    #       Dx - Dimension of Latent Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPXX - Second moment of P w.r.t. X
    #       Tol - Tolerance for Gradient Descent
    #
    # Outputs:
    #       VarMarg - Implicit Likelihood Variational Marginal Entropy
    ###############################################################################
    mux0 = np.zeros((Dx,1))
    chol_sigmax0 = np.eye(Dx) 
    Margx0 = np.concatenate((mux0.flatten(),chol_sigmax0.flatten()))
    
    conditions = np.eye(Dx)
    arr = -1*np.ones((Dx,Dx))
    conditions = conditions+np.triu(arr, 1)
    MargBound = []
    for i in range(len(mux0.flatten())):
        MargBound.append((-np.inf,np.inf))
    for j in range(len(conditions.flatten())):
        if (conditions==1).flatten()[j]:
            MargBound.append((.005,np.inf))
        elif(conditions==-1).flatten()[j]:
            MargBound.append((0,0))
        else:
            MargBound.append((-np.inf,np.inf))
    if FullOut == True:
        history = []
        def callback(x):
            fobj = GDMarg(x,Dx,EPX,EPXX)
            history.append(fobj)
        VarMarg = scipy.optimize.minimize(fun=GDMarg, x0=Margx0, args=(Dx,EPX,EPXX),bounds=tuple(MargBound), jac=MargDerivative,method='L-BFGS-B',tol=Tol, callback=callback) 
        VarMarg=history
    else:
        VarMarg = scipy.optimize.minimize(fun=GDMarg, x0=Margx0, args=(Dx,EPX,EPXX),bounds=tuple(MargBound), jac=MargDerivative,method='L-BFGS-B', tol=Tol) 
        VarMarg=VarMarg.fun
    return VarMarg

def GDMarg(params,Dx,EPX,EPXX):
    ###############################################################################
    # Outline: Implicit Likelihood Variational Approximation optimization function
    #
    # Inputs:
    #       params - Parameters of variational distribution [mu_x,chol_Sigma_x]
    #       Dx - Dimension of Latent Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPXX - Second moment of P w.r.t. X
    #
    # Outputs:
    #       Marg - Marginal Entropy evaluation
    ###############################################################################
    muq_x = params[0:Dx].reshape(Dx,1)
    
    chol_Sigmaq_x = params[Dx:]
    chol_Sigmaq_x = chol_Sigmaq_x.reshape(Dx,Dx)
    
    Sigmaq_x = np.matmul(chol_Sigmaq_x,chol_Sigmaq_x.T)
    
    Marg = EvalMarg(Dx,EPX,EPXX,muq_x,Sigmaq_x)
    return Marg.flatten()

def EvalMarg(Dx,EPX,EPXX,muq_x,Sigmaq_x):
    ###############################################################################
    # Outline: Evaluates the Variational Marginal Entropy
    #
    # Inputs:
    #       Dx - Dimension of Latent Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPXX - Second moment of P w.r.t. X
    #       muq_x - Mean of Latent Variable in the Variational Distribution
    #       Sigmaq_x -  Variance of Latent Variable in Variational Distribution
    #
    # Outputs:
    #       Marg - Marginal Entropy evaluation
    ###############################################################################
    Sigmaq_x_inv = np.linalg.inv(Sigmaq_x)
    Marg  = .5*(Dx*np.log(2*np.pi)+np.log(np.linalg.det(Sigmaq_x))+\
        np.trace(np.matmul(Sigmaq_x_inv,(EPXX-np.outer(EPX,EPX))))+\
        np.matmul(EPX.T,np.matmul(Sigmaq_x_inv,EPX))-2*np.matmul(muq_x.T,np.matmul(Sigmaq_x_inv,EPX))+\
        np.matmul(muq_x.T,np.matmul(Sigmaq_x_inv,muq_x)))
    return Marg[0][0]

def MargDerivative(params,Dx,EPX,EPXX):
    ###############################################################################
    # Outline: Evaluates the Derivative Variational Marginal Entropy
    #
    # Inputs:
    #       params - Parameters of variational distribution [mu_x,chol_Sigma_x]
    #       Dx - Dimension of Latent Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPXX - Second moment of P w.r.t. X
    #
    # Outputs:
    #       dparams - derivative of each parameter
    ###############################################################################
    muq_x = tf.Variable(tf.convert_to_tensor(params[0:Dx].reshape(Dx,1)), name='muq_x')
    chol_Sigmaq_x = tf.Variable(tf.convert_to_tensor(params[Dx:].reshape(Dx,Dx)), name='chol_Sigmaq_x')
    EPX = EPX.reshape((len(EPX),1))
    with tf.GradientTape(persistent=True) as tape:
        Sigmaq_x = tf.linalg.matmul(chol_Sigmaq_x,tf.transpose(chol_Sigmaq_x))
        Sigmaq_x_inv = tf.linalg.inv(Sigmaq_x)        
        Marg  = .5*(Dx*tf.math.log(2*np.pi)+tf.math.log(tf.linalg.det(Sigmaq_x))+\
            tf.linalg.trace(tf.linalg.matmul(Sigmaq_x_inv,(EPXX-np.outer(EPX,EPX))))+\
            tf.linalg.matmul(EPX.T,tf.linalg.matmul(Sigmaq_x_inv,EPX))-2*tf.linalg.matmul(tf.transpose(muq_x),tf.linalg.matmul(Sigmaq_x_inv,EPX))+\
            tf.linalg.matmul(tf.transpose(muq_x),tf.linalg.matmul(Sigmaq_x_inv,muq_x)))
    [dmu, dSigmachol] = tape.gradient(Marg, [muq_x, chol_Sigmaq_x])
    dmu = tf.make_ndarray(tf.make_tensor_proto(dmu))
    dSigmachol = tf.make_ndarray(tf.make_tensor_proto(dSigmachol))
    dparams = np.concatenate((dmu.flatten(),dSigmachol.flatten()))
    return dparams

def CondEntGradientDescent(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,Tol,FullOut=False):
    ###############################################################################
    # Outline: Calculates Implicit Likelihood Variational Conditional Entropy by 
    #          Gradient Descent
    #
    # Inputs:
    #       Dx - Dimension of Latent Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Cross Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #       Tol - Tolerance for Gradient Descent
    #
    # Outputs:
    #       VarCond.fun - Implicit Likelihood Variational Conditional Entropy
    ###############################################################################
    b = np.zeros((Dx,1))
    A = np.matmul((EPXY-np.outer(EPX,EPY)),np.linalg.inv(EPYY-np.outer(EPY,EPY)))
    Sigma = np.eye(Dx)
    Condx0 = np.concatenate((np.concatenate((A.flatten(),b.flatten())),Sigma.flatten()))
    
    conditions = np.eye(Dx)
    arr = -1*np.ones((Dx,Dx))
    conditions = conditions+np.triu(arr, 1)
    CondBound = []
    for i in range(len(A.flatten())+len(b.flatten())):
        CondBound.append((-np.inf,np.inf))

    for j in range(len(conditions.flatten())):
        if (conditions==1).flatten()[j]:
            CondBound.append((0.005,np.inf))
        elif(conditions==-1).flatten()[j]:
            CondBound.append((0,0))
        else:
            CondBound.append((-np.inf,np.inf))
            #(EPXX-np.outer(EPX,EPX))-np.matmul((EPXY-np.outer(EPX,EPY)),np.matmul(EPYY-np.outer(EPY,EPY),((EPXY-np.outer(EPX,EPY)).T)))
    if FullOut == True:
        history = []
        def callback(x):
            fobj = GDCond(x,Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY)
            history.append(fobj)
        VarCond = scipy.optimize.minimize(fun=GDCond, x0=Condx0, args=(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY),bounds=tuple(CondBound),jac=CondDerivative, method='L-BFGS-B', tol=Tol,callback=callback)
        VarCond=history
    else:
        VarCond = scipy.optimize.minimize(fun=GDCond, x0=Condx0, args=(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY),bounds=tuple(CondBound),jac=CondDerivative, method='L-BFGS-B', tol=Tol)
        VarCond = VarCond.fun
    return VarCond

def GDCond(params, Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY):
    ###############################################################################
    # Outline: Implicit Likelihood Variational Approximation optimization function
    #
    # Inputs:
    #       params - Parameters of varaitional distribution [mu_x,mu_y,chol_Sigma_joint]
    #       Dx - Dimension of Latent Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Cross Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #
    # Outputs:
    #       Cond - Conditional Entropy evaluation
    ###############################################################################

    A = params[0:Dx*Dy].reshape(Dx,Dy)
    b = params[Dx*Dy:(Dx*Dy+Dx)].reshape(Dx,1)
    
    chol_Sigmaq_joint = params[(Dx*Dy+Dx):]
    chol_Sigmaq_joint = chol_Sigmaq_joint.reshape((Dx),(Dx))

    Sigmaq = np.matmul(chol_Sigmaq_joint,chol_Sigmaq_joint.T)
    
    Cond = EvalCond(Dx,EPX,EPY,EPXX,EPXY,EPYY,A,b,Sigmaq)
    return Cond.flatten()

def EvalCond(Dx,EPX,EPY,EPXX,EPXY,EPYY,A,b,Sigmaq):
    ###############################################################################
    # Outline: Evaluates the Variational Conditional Entropy
    #
    # Inputs:
    #       Dx - Dimension of Latent Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Ctoss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #       A, b - mean of conditional is Ay+b
    #       Sigmaq - Covariance of conditional distribution
    #
    # Outputs:
    #       Cond - Conditional Entropy evaluation
    ###############################################################################
    Sigmaq_inv = np.linalg.inv(Sigmaq)
    A1 = np.matmul(Sigmaq_inv,A)
    B1 = np.matmul(A.T,A1)
    EPX = EPX.reshape((len(EPX),1))
    EPY = EPY.reshape((len(EPY),1))
    
    Cond  = .5*(Dx*np.log(2*np.pi)+np.log(np.linalg.det(Sigmaq))+\
        np.trace(np.matmul(Sigmaq_inv,EPXX))-\
        2*(np.matmul(b.T,np.matmul(Sigmaq_inv,EPX)) +\
        np.trace(np.matmul(A1,EPXY.T)))+\
        np.trace(np.matmul(B1,EPYY))+\
        2*np.matmul(b.T,np.matmul(A1,EPY))+np.matmul(b.T,np.matmul(Sigmaq_inv,b)))

    return Cond[0][0]

def CondDerivative(params, Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY):
    ###############################################################################
    # Outline: Evaluates the Derivative Variational Conditional Entropy
    #
    # Inputs:
    #       params - Parameters of varaitiona distribution [mu_x,mu_y,chol_Sigma_joint]
    #       Dx - Dimension of Latenat Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #
    # Outputs:
    #       dparams - derivative of each parameter
    ###############################################################################
    A = tf.Variable(tf.convert_to_tensor(params[0:Dx*Dy].reshape(Dx,Dy)), name='A')
    b = tf.Variable(tf.convert_to_tensor(params[Dx*Dy:(Dx*Dy+Dx)].reshape(Dx,1)), name='b')
    
    chol_Sigmaq_joint = tf.Variable(tf.convert_to_tensor(params[(Dx*Dy+Dx):].reshape(Dx,Dx)), name='chol_Sigmaq_joint')
    EPX = EPX.reshape((len(EPX),1))
    EPY = EPY.reshape((len(EPY),1))
    
    with tf.GradientTape(persistent=True) as tape:
        Sigmaq = tf.linalg.matmul(chol_Sigmaq_joint,tf.transpose(chol_Sigmaq_joint))
        Sigmaq_inv = tf.linalg.inv(Sigmaq)
        A1 =tf.linalg.matmul(Sigmaq_inv,A)
        B1 = tf.linalg.matmul(tf.transpose(A),A1)
        
        Cond  = .5*(Dx*tf.math.log(2*np.pi)+tf.math.log(tf.linalg.det(Sigmaq))+\
            tf.linalg.trace(tf.linalg.matmul(Sigmaq_inv,(EPXX)))-\
            2*(tf.linalg.matmul(tf.transpose(b),tf.linalg.matmul(Sigmaq_inv,EPX)) +\
            tf.linalg.trace(tf.linalg.matmul(A1,tf.transpose(EPXY))))+\
            tf.linalg.trace(tf.linalg.matmul(B1,(EPYY)))+\
            2*tf.linalg.matmul(tf.transpose(b),tf.linalg.matmul(A1,EPY))+tf.linalg.matmul(tf.transpose(b),tf.linalg.matmul(Sigmaq_inv,b)))

    [dA,db, dSigmachol] = tape.gradient(Cond, [A, b,chol_Sigmaq_joint])
    dA = tf.make_ndarray(tf.make_tensor_proto(dA))
    db = tf.make_ndarray(tf.make_tensor_proto(db))
    dSigmachol = tf.make_ndarray(tf.make_tensor_proto(dSigmachol))
    dparams = np.concatenate((dA.flatten(),np.concatenate((db.flatten(),dSigmachol.flatten()))))
    return dparams

def MIOpt(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,const,Tol):
    ###############################################################################
    # Outline: Calculates an Optimal Variational Distribution that matches the 
    #          True Distributions Mutual Information (NOT USED IN PAPER)
    #
    # Inputs:
    #       Dx - Dimension of Latenat Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #       const - value of True MI
    #       Tol - Tolerance for Gradient Descent
    #
    # Outputs:
    #       mux - Mean of Latent Variable of Variational Distribution
    #       muy - Mean of Observation Variable of Variational Distribution
    #       Sigmax - Variance of Latent Variable of Variational Distribution
    #       Sigmaxy - Covariance of Varaitional Distribution 
    #       Sigmay - Variance of Latent Variable of Variational Distribution
    ###############################################################################
    mux0_joint = np.zeros((Dx+Dy,1))
    sigmax0_joint = np.zeros((Dx+Dy,Dx+Dy))

    mux0_joint[0:Dx] = EPX.reshape((Dx,1))
    mux0_joint[Dx:(Dx+Dy)] = EPY.reshape((Dy,1))
    sigmax0_joint[0:Dx,0:Dx] = EPXX-np.outer(EPX,EPX)
    sigmax0_joint[0:Dx,Dx:(Dx+Dy)] = .9*(EPXY-np.outer(EPX,EPY))
    sigmax0_joint[Dx:(Dx+Dy),0:Dx] = .9*(EPXY-np.outer(EPX,EPY)).T
    sigmax0_joint[Dx:(Dx+Dy),Dx:(Dx+Dy)] = EPYY-np.outer(EPY,EPY)
    chol_sigmax0_joint = np.linalg.cholesky(sigmax0_joint)

    Condx0 = np.concatenate((mux0_joint.flatten(),chol_sigmax0_joint.flatten()))
    
    conditions = np.eye(Dx+Dy)
    arr = -1*np.ones((Dx+Dy,Dx+Dy))
    conditions = conditions+np.triu(arr, 1)
    CondBound = []
    for i in range(len(mux0_joint.flatten())):
        CondBound.append((-np.inf,np.inf))

    for j in range(len(conditions.flatten())):
        if (conditions==1).flatten()[j]:
            CondBound.append((0.01,np.inf))
        elif(conditions==-1).flatten()[j]:
            CondBound.append((0,0))
        else:
            CondBound.append((-np.inf,np.inf))
            
    VarCond = scipy.optimize.minimize(fun=GDMIOPT, x0=Condx0, args=(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,const),bounds=tuple(CondBound),jac=GDMIOPTDerivs, method='L-BFGS-B', tol=Tol)
    mux = VarCond.x[0:Dx].reshape(Dx,1)
    muy = VarCond.x[Dx:(Dx+Dy)].reshape(Dy,1)
    cholSigma = VarCond.x[(Dx+Dy):].reshape((Dx+Dy),(Dx+Dy))
    Sigma = np.matmul(cholSigma,cholSigma.T)
    Sigmax = Sigma[0:Dx,0:Dx]
    Sigmaxy = Sigma[0:Dx,Dx:(Dx+Dy)]
    Sigmay = Sigma[Dx:(Dx+Dy),Dx:(Dx+Dy)]
    return mux,muy,Sigmax,Sigmaxy,Sigmay

def GDMIOPT(params, Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,const):
    ###############################################################################
    # Outline: Evaluates the Absolute Error of the Difference of True and 
    #          Variational MI (NOT USED IN PAPER)
    #
    # Inputs:
    #       params - Parameters of varaitiona distribution [mu_x,mu_y,chol_Sigma_joint]
    #       Dx - Dimension of Latenat Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #       const - value of True MI
    #
    # Outputs:
    #       MIN - Value of difference of True and Variational MI
    ###############################################################################
    muq_x = params[0:Dx].reshape(Dx,1)
    muq_y = params[Dx:(Dx+Dy)].reshape(Dy,1)

    chol_Sigmaq_joint = params[(Dx+Dy):]
    chol_Sigmaq_joint = chol_Sigmaq_joint.reshape((Dx+Dy),(Dx+Dy))
    
    Sigmaq_joint = np.matmul(chol_Sigmaq_joint,chol_Sigmaq_joint.T)
    Sigmaq_x = Sigmaq_joint[0:Dx,0:Dx]
    Sigmaq_xy = Sigmaq_joint[0:Dx,Dx:(Dx+Dy)]
    Sigmaq_y = Sigmaq_joint[Dx:(Dx+Dy),Dx:(Dx+Dy)]
    
    OptMarg = EvalMarg(Dx,EPX,EPXX,muq_x,Sigmaq_x)
    OptCond = EvalCond(Dx,EPX,EPY,EPXX,EPXY,EPYY,muq_x,muq_y,Sigmaq_x,Sigmaq_xy,Sigmaq_y)
    
    MI = OptMarg-OptCond
    MIN = np.abs(MI-const)
    return MIN.flatten()

def GDMIOPTDerivs(params, Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,const):
    ###############################################################################
    # Outline: Evaluates the Derivative of Absolute Error of the Difference of True 
    #          and Variational MI (NOT USED IN PAPER)
    #
    # Inputs:
    #       params - Parameters of varaitiona distribution [mu_x,mu_y,chol_Sigma_joint]
    #       Dx - Dimension of Latenat Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #       const - value of True MI
    #
    # Outputs:
    #       dparams - derivative of each parameter
    ###############################################################################
    muq_x = params[0:Dx].reshape(Dx,1)
    muq_y = params[Dx:(Dx+Dy)].reshape(Dy,1)

    chol_Sigmaq_joint = params[(Dx+Dy):]
    chol_Sigmaq_joint = chol_Sigmaq_joint.reshape((Dx+Dy),(Dx+Dy))
    
    Sigmaq_joint = np.matmul(chol_Sigmaq_joint,chol_Sigmaq_joint.T)
    Sigmaq_x = Sigmaq_joint[0:Dx,0:Dx]
    Sigmaq_xy = Sigmaq_joint[0:Dx,Dx:(Dx+Dy)]
    Sigmaq_y = Sigmaq_joint[Dx:(Dx+Dy),Dx:(Dx+Dy)]
    
    OptMarg = EvalMarg(Dx,EPX,EPXX,muq_x,Sigmaq_x)
    OptCond = EvalCond(Dx,EPX,EPY,EPXX,EPXY,EPYY,muq_x,muq_y,Sigmaq_x,Sigmaq_xy,Sigmaq_y)
    
    params1 = np.concatenate((params[0:Dx],chol_Sigmaq_joint[0:Dx,0:Dx].flatten()))
    MargDeriv = MargDerivative(params1,Dx,EPX,EPXX)
    CondDeriv = CondDerivative(params, Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY)
    
    MI = OptMarg-OptCond
    MIN = np.sign(MI-const).flatten()
    dmu_x = MIN*(MargDeriv[0:Dx]-CondDeriv[0:Dx])
    dmu_y = -1*MIN*(CondDeriv[Dx:(Dx+Dy)])
    dSigmachol = -1*MIN*(CondDeriv[(Dx+Dy):]).reshape((Dx+Dy),(Dx+Dy))
    dSigmachol[0:Dx,0:Dx] += MIN*(MargDeriv[Dx:]).reshape((Dx,Dx))
    dparams = np.concatenate((dmu_x.flatten(),np.concatenate((dmu_y.flatten(),dSigmachol.flatten()))))
    return dparams

def MeanAndVariance(N,K,MI):
    ###############################################################################
    # Outline: Calculate the Mean and Variane of a run
    #
    # Inputs:
    #       N - Number of Different Sample Sizes
    #       K - Number of Itterations
    #       MI - Values of Mutual Informations [Storage of Itteration, Storage of Sample Size, 0=MI 1=Run Time]
    #
    # Outputs:
    #       MIMean - Mean at each Sample Size [0=MI 1=Run Time]
    #       MIVariance - Varaince of each Sample Size [0=MI 1=Run Time]
    ###############################################################################
    MIMean = np.zeros((N,2))
    MIVariance = np.zeros((N,2))

    MIMean[:,0] = sum(MI[:,:,0])/K
    MIMean[:,1] = sum(MI[:,:,1])/K

    MIVariance[:,0] = sum((MI[:,:,0]-sum(MI[:,:,0])/K)**2)/(K-1)
    MIVariance[:,1] = sum((MI[:,:,1]-sum(MI[:,:,1])/K)**2)/(K-1)
    return MIMean, MIVariance

def confidence_ellipse(mean, cov, ax, n_std=1, **kwargs):
    ###############################################################################
    # Outline: Adds standard deviation ellipse to plot
    #
    # Inputs:
    #       mean - Mean of Gaussian Distribution to Plot
    #       cov - Covariance of Gaussian Distribution to Plot
    #       ax - Plot to add standard deviation
    #       n_std - number of standard of deviations to plot
    #
    # Outputs:
    #       pearson - correlation to distribution. output is unimportant for future use
    ###############################################################################
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1,1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    return pearson

if __name__ == "__main__":

    # random.seed(10)
    PRINTRESULTS = True # Prints all the values of MI calculated
    PLOTEXAMPLE = True # Plots samples of GMM with learned Var MM on top
    PlotPreSaved = True # This is for plots in Figure 2
    Oracle = False # Not used in this paper.

    ###################### Create GMM Model ######################################
    K = 5 # Number of Trials
    Ns = [400] #,500,1000,2000,5000Number of Samples
    Tol = 10**(-6) 
    M = 5 # Number of Componentes
    Dx = 60 # Dimension of X 
    Dy = 5 # Dimension of Y
    ws,mus,Sigmas = GaussianMixtureParams(M,Dx,Dy) # Generates Parameters for GMM   
    
    ################# Pre-Saved GMM Distributions for Fig 2 ######################
    if PlotPreSaved:   
        # Dx = 1
        # Dy = 1
        # M = 2
        # K=1
        # ws = np.array([.5,.5])
        # mus = [np.array([[5],[5]]),np.array([[-5],[-5]])]#[np.array([[3],[2]]),np.array([[4.5],[-5]])]
        # Sigmas = [.4*np.eye(2),.2*np.eye(2)]
  
        # Dx = 1
        # Dy = 1
        # M = 2
        # K=1
        # ws = np.array([.2,.8])
        # mus = [np.array([[1],[3.5]]),np.array([[-5],[4.5]])]#[np.array([[3],[2]]),np.array([[4.5],[-5]])]
        # Sigmas = [.7*np.eye(2),.1*np.eye(2)]
        
        Dx = 1
        Dy = 1
        M = 2
        K=1
        ws = np.array([0.2, 0.8])
        mus = [np.array([[-1],[1]]),np.array([[2],[-1]])]
        Sigmas = [np.array([[1, 0.5], [0.5, 1]]), np.array([[1.5, -.5], [-.5, 1.5]])]
    #############################################################################
    
    SampleMI = np.zeros((K,len(Ns),2))
    MMMI = np.zeros((K,len(Ns),2))
    GDMI = np.zeros((K,len(Ns),2))
    BAMI = np.zeros((K,len(Ns),2))
    VMMI = np.zeros((K,len(Ns),2))
    BAGD = np.zeros((K,len(Ns),2))
    VMGD = np.zeros((K,len(Ns),2))
    Mine = np.zeros((K,len(Ns),2))
    n=-1
    for N in Ns: # Loop over Number of Samples
        n+=1
        # print(N)
        for k in range(K): # Loop over number or itterations
            sample = SampleGMM(N,ws,mus,Sigmas)           
            
            MomentsTime0 = time.time()
            EPX, EPY, EPXX, EPXY, EPYY = pmoments(sample,Dx,Dy)
            MomentsTime1 = time.time()
            MomentsTime = MomentsTime1-MomentsTime0
            
            ### IGNORE THIS LINE, THE FIRST COMPUTATION OF CONDENT ALWAYS HAS A MISTIMING
            MMCondEnt = CondEntMomentMatch(Dx, EPX, EPY, EPXX, EPXY, EPYY)
            ########################################################################
            
            MMMargTime0 = time.time()
            MMMargEnt = MargEntMomentMatch(Dx, EPX, EPXX)
            MMMargTime1 = time.time()
            MMMargTime = MMMargTime1 - MMMargTime0
            
            
            MMCondTime0 = time.time()
            MMCondEnt = CondEntMomentMatch(Dx, EPX, EPY, EPXX, EPXY, EPYY)
            MMCondTime1 = time.time()
            MMCondTime = MMCondTime1 - MMCondTime0
            
            SampleMargTime0 = time.time()
            SampleMargEnt = MargEntGMM(sample,Dx,ws,mus,Sigmas)
            SampleMargTime1 = time.time()
            SampleMargTime = SampleMargTime1 - SampleMargTime0
            
            SampleCondTime0 = time.time()
            SampleCondEnt = CondEntGMM(sample,Dx,Dy,ws,mus,Sigmas)
            SampleCondTime1 = time.time()
            SampleCondTime = SampleCondTime1 - SampleCondTime0
            
            GDMargTime0 = time.time()
            GDMargEnt = MargEntGradientDescent(Dx,EPX,EPXX,Tol,FullOut=True)
            GDMargTime1 = time.time()
            GDMargTime = GDMargTime1 - GDMargTime0
            
            GDCondTime0 = time.time()
            GDCondEnt = CondEntGradientDescent(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,Tol,FullOut=True)
            GDCondTime1 = time.time()
            GDCondTime = GDCondTime1-GDCondTime0
            
            ############### Sample MI ################################
            SampleMI[k,n,0] = SampleMargEnt-SampleCondEnt
            SampleMI[k,n,1] = SampleMargTime+SampleCondTime+.00015
            ##########################################################

            # ################### MM IL ################################
            MMMI[k,n,0] = MMMargEnt-MMCondEnt
            MMMI[k,n,1] = MomentsTime+MMMargTime+MMCondTime+.00015
            # ##########################################################
            
            ################### GD IL ################################
            GDMI[k,n,0] = GDMargEnt[-1]-GDCondEnt[-1]
            GDMI[k,n,1] = GDMargTime+GDCondTime+.00015
            ##########################################################
            
            # ################### MM VM ################################
            VMMI[k,n,0] = MMMargEnt-SampleCondEnt
            VMMI[k,n,1] = MomentsTime+MMMargTime+SampleCondTime+.00015
            # ##########################################################
            
            ################### GD VM ################################
            VMGD[k,n,0] = GDMargEnt[-1]-SampleCondEnt
            VMGD[k,n,1] = GDMargTime+SampleCondTime+.00015
            ##########################################################
            
            # ################### MM BA ################################
            BAMI[k,n,0] = SampleMargEnt-MMCondEnt
            BAMI[k,n,1] = SampleMargTime+MomentsTime+MMCondTime+.00015
            # ##########################################################
            
            ################### GD BA ################################
            BAGD[k,n,0] = SampleMargEnt-GDCondEnt[-1]
            BAGD[k,n,1] = SampleMargTime+GDCondTime+.00015
            ##########################################################
            
            #################### MINE ################################
            MineTime0 = time.time()
            data = (sample[:,:Dx],sample[:,Dx:])
            model = FullyConnected(var1_dim=Dx, var2_dim=Dy, L=1, H=[10])
            mine_obj = MINE(model, data, lr=5 * 1e-4)

            mine_obj.train(n_epoch=5000, batch_size=N)
            Mine[k,n,0] =np.mean(mine_obj.train_lb[-100:])
            MineTime1 = time.time()
            Mine[k,n,1] = MineTime1-MineTime0+.00015
            ##########################################################
           
############################# Plots 3b in paper ##################################
            if N==Ns[-1] and not PlotPreSaved:
                if k==0 or k==1:
                    if len(GDCondEnt)>len(GDMargEnt):
                        for j in range(len(GDCondEnt)-len(GDMargEnt)):
                            GDMargEnt.append(GDMargEnt[-1])
                    else:
                        for j in range(len(GDMargEnt)-len(GDCondEnt)):
                            GDCondEnt.append(GDCondEnt[-1])

                    GDMIFull = np.asarray(GDMargEnt)-np.asarray(GDCondEnt)
                    MMMIFull = (MMMargEnt-MMCondEnt)*np.ones((len(GDCondEnt),1))
                    fig2 = go.Figure([
                        go.Scatter(
                            x=np.linspace(0,len(GDCondEnt)-1,len(GDCondEnt)),
                            y=MMMIFull.flatten(),
                            line=dict(color='rgb(0,100,80)', width=2),
                            mode='lines',
                            name='I True'
                        ),
                        go.Scatter(
                            x=np.linspace(0,len(GDCondEnt)-1,len(GDCondEnt)),
                            y=MMMIFull.flatten(),
                            line=dict(color='rgb(0,0,0)', width=2),
                            mode='lines',
                            name='MINE'
                        ),
                        go.Scatter(
                            x=np.linspace(0,len(GDCondEnt)-1,len(GDCondEnt)),
                            y=MMMIFull.flatten(),
                            line=dict(color='rgb(0,0,255)', width=2),
                            mode='lines',
                            name='I<sub>post</sub> MM'
                        ),go.Scatter(
                            x=np.linspace(0,len(GDCondEnt)-1,len(GDCondEnt)),
                            y=MMMIFull.flatten(),
                            line=dict(color='rgb(0,0,255)', width=2,dash='dot'),
                            mode='lines',
                            name='I<sub>post</sub> GD'
                        ),go.Scatter(
                            x=np.linspace(0,len(GDCondEnt)-1,len(GDCondEnt)),
                            y=MMMIFull.flatten(),
                            line=dict(color='rgb(180,0,255)', width=2),
                            mode='lines',
                            name='I<sub>marg</sub> MM'
                        ),go.Scatter(
                            x=np.linspace(0,len(GDCondEnt)-1,len(GDCondEnt)),
                            y=MMMIFull.flatten(),
                            line=dict(color='rgb(180,0,255)', width=2, dash='dot'),
                            mode='lines',
                            name='I<sub>marg</sub> GD'
                        ),
                        go.Scatter(
                            x=np.linspace(0,len(GDCondEnt)-1,len(GDCondEnt)),
                            y=MMMIFull.flatten(),
                            line=dict(color='rgb(255,0,0)', width=3),
                            mode='lines',
                            name='I<sub>m+p</sub> MM'
                        ),
                        go.Scatter(
                            x=np.linspace(0,len(GDCondEnt)-1,len(GDCondEnt)),
                            y=GDMIFull.flatten(),
                            line=dict(color='rgb(255,0,0)', width=3, dash='dot'),
                            mode='lines',
                            name='I<sub>m+p</sub> GD'
                        )])
                    fig2.update_xaxes(title_text="Gradient Step", type="log",dtick ="D2")
                    fig2.update_layout(
                        xaxis={
                            "tickmode": "array",
                            "tickvals": pd.to_numeric(
                                [f"{n:.1g}" for n in np.array([2,5,10,20,50,100,200])]
                            ),
                        }
                    )
                    fig2.update_yaxes(title_text="MI Approximation", type="log",dtick ="D2")
                    fig2.update_layout(
                        yaxis={
                            "tickmode": "array",
                            "tickvals": pd.to_numeric(
                                [f"{n:.1g}" for n in np.array([2,5,10,20,50,100,200])]
                            ),
                        }
                    )
                    fig2.update_layout(legend=dict(itemsizing='constant'))
                    fig2.update_layout(font=dict(size=25))
                    fig2.update_layout(plot_bgcolor='white')
                    fig2.update_xaxes(
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        linecolor='black',
                        gridcolor='lightgrey'
                    )
                    fig2.update_yaxes(
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        linecolor='black',
                        gridcolor='lightgrey'
                    )
                    fig2.show()
                    # fig2.write_image("LargeGMMGDStepNew.pdf")
################################################################################## 

    ########################### Oracle Distribution#######################################
    ### Uses the True MI to find a varaitaional distribution that matches the Mutual
    ### Information Exactly. This is to comapare the best possible variational
    ### distribution to the moment matched distribution. Not used in paper
    ######################################################################################
    if Oracle:

        EPX, EPY, EPXX, EPXY, EPYY = pmoments(sample,Dx,Dy)
        const  = SampleMI[k,n,0]
        mux,muy,Sigmax,Sigmaxy,Sigmay = MIOpt(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,const,10**(-20))
        VarMean = np.append(mux.flatten(),muy.flatten()).reshape(Dx+Dy,1)
        VarVar = np.append(np.append(np.append(Sigmax.flatten(),Sigmaxy.flatten()),(Sigmaxy.T).flatten()),Sigmay.flatten()).reshape(Dx+Dy,Dx+Dy)
        
        CondSigma = Sigmax-np.matmul(Sigmaxy,np.matmul(np.linalg.inv(Sigmay),Sigmaxy.T))
        
        OptMarg = EvalMarg(Dx,EPX,EPXX,mux,Sigmax)
        OptCond = EvalCond(Dx,EPX,EPY,EPXX,EPXY,EPYY,mux,muy,Sigmax,Sigmaxy,Sigmay)
        OptMI = OptMarg-OptCond
        
        MMMean = np.append(EPX.flatten(),EPY.flatten()).reshape(Dx+Dy,1)
        MMVar = np.append(np.append(np.append((EPXX-np.outer(EPX,EPX)).flatten(),(EPXY-np.outer(EPX,EPY)).flatten()),((EPXY-np.outer(EPX,EPY)).T).flatten()),(EPYY-np.outer(EPY,EPY)).flatten()).reshape(Dx+Dy,Dx+Dy)

        if PLOTEXAMPLE:
            fig2, ax = plt.subplots()
            ax.scatter(sample[:,0], sample[:,1], c='black')
            p1 = confidence_ellipse(VarMean, VarVar, ax, n_std=1, facecolor='none', edgecolor='springgreen',label='Oracle')
            p2 = confidence_ellipse(MMMean, MMVar, ax, n_std=1, facecolor='none', edgecolor='red',label='MM')
            ax.legend(fontsize=20)
            plt.show()
    ###############################################################################
    
    if PlotPreSaved:
        def multivariate_gaussian_pdf(x, mean, cov):
            """Multivariate Gaussian PDF"""
            k = mean.shape[0]
            det = np.linalg.det(cov)
            inv = np.linalg.inv(cov)
            norm = 1.0 / np.sqrt((2*np.pi)**k * det)
            exp = np.exp(-0.5 * np.dot(np.dot((x-mean).T, inv), x-mean))
            return norm * exp

        def gmm_pdf(x, weights, means, covs):
            """Gaussian Mixture Model PDF"""
            return np.sum(np.fromiter((weights[i] * multivariate_gaussian_pdf(x, means[i], covs[i])
                                    for i in range(len(weights))), dtype=float))

        def gaussian_marginal_pdf(x, mean, cov):
            """Marginal PDF of the Gaussian with respect to the first variable"""
            return multivariate_normal.pdf([x, 0], mean=[mean[0], 0], cov=[[cov[0,0], 0], [0, 1]])

        def gmm_marginal_pdf(x, weights, means, covs):
            """Marginal PDF of the GMM with respect to the first variable"""
            pdf_values = np.array([weights[i] * gaussian_marginal_pdf(x, means[i], covs[i]) for i in range(len(weights))])
            return np.sum(pdf_values)
        
        # Define the GMM and Gaussian parameters
        ws = [0.2, 0.8]
        mus = np.array([[-1, 1], [2, -1]])
        Sigmas = np.array([[[1, 0.5], [0.5, 1]], [[1.5, -.5], [-.5, 1.5]]])

        gaussian_mean = np.array([EPX.flatten()[0], EPY.flatten()[0]])
        gaussian_cov = np.array([[(EPXX-np.outer(EPX,EPX)).flatten()[0], (EPXY-np.outer(EPX,EPY)).flatten()[0]], [(EPXY-np.outer(EPX,EPY)).flatten()[0], (EPYY-np.outer(EPY,EPY)).flatten()[0]]])

        # Define a grid to evaluate the density on
        x_min, x_max = -3, 5
        y_min, y_max = -3, 3
        xx, yy = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Evaluate the density on the grid
        gmm_z = np.array([gmm_pdf(x, ws, mus, Sigmas) for x in grid])
        gmm_z = gmm_z.reshape(xx.shape)

        gaussian_z = np.array([multivariate_gaussian_pdf(x, gaussian_mean, gaussian_cov) for x in grid])
        gaussian_z = gaussian_z.reshape(xx.shape)

        gmm_marginal_z = np.array([gmm_marginal_pdf(x, ws, mus, Sigmas) for x in xx[:, 0]])
        gaussian_marginal_z = np.array([gaussian_marginal_pdf(x, gaussian_mean, gaussian_cov) for x in xx[:, 0]])

        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['text.usetex'] = True
        
        # Plot the contour plot of the GMM
        fig2, ax2 = plt.subplots()
        cntr1 = ax2.contour(xx, yy, gmm_z, levels=np.array([.001,.005,.01,.02,.03,.05,.08]), colors='k')
        cntr2 = ax2.contour(xx, yy, gaussian_z, levels=np.array([.002,.01,.03,.05,.07]), colors='red')
        h1,_ = cntr1.legend_elements()
        h2,_ = cntr2.legend_elements()
        ax2.legend([h1[0], h2[0]], ['GMM', 'MM'],fontsize=30,loc='upper right')
        plt.axis("tight")
        plt.savefig("GMMContourCameraReady.pdf")
        plt.show()
        
        # Plot the contour plot of the GMM and Gaussian, as well as their marginal PDFs
        fig1, ax1 = plt.subplots()
        dens1 = ax1.plot(xx[:, 0], gmm_marginal_z, color='k')
        dens2 = ax1.plot(xx[:, 0], gaussian_marginal_z, color='red')
        plt.savefig("GMMpdfCameraReady.pdf")
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        Methods = ['$I_{post}$',  '$I_{m+p}$','$I$', '$I_{marg}$']
        MIs = [BAMI[k,n,0],MMMI[k,n,0],SampleMI[k,n,0],VMMI[k,n,0]]
        ax.bar(Methods,MIs)
        ax.set_ylabel('Mutual Information',fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.savefig('GMMbarOrderedCameraReady.pdf', bbox_inches='tight')
        plt.show()

    ########################## Debugging Prints ###################################
    if PRINTRESULTS:
        print("Sample MI:       %s" %SampleMI[k,n,0])
        print("Moment Match MI: %s" %MMMI[k,n,0])
        # print("Oracle MI:       %s" %OptMI)
        print("BA MI:           %s" %BAMI[k,n,0])
        print("VM MI:           %s" %VMMI[k,n,0])
    ############################################################################### 


########################## Plot Result ############################################
    if PLOTEXAMPLE:    
        SampleMIMean, SampleMIVariance = MeanAndVariance(len(Ns),K,SampleMI)
        MMMIMean, MMMIVariance = MeanAndVariance(len(Ns),K,MMMI)
        GDMIMean, GDMIVariance = MeanAndVariance(len(Ns),K,GDMI)
        BAMIMean, BAMIVariance = MeanAndVariance(len(Ns),K,BAMI)
        VMMIMean, VMMIVariance = MeanAndVariance(len(Ns),K,VMMI)
        BAGDMean, BAGDVariance = MeanAndVariance(len(Ns),K,BAGD)
        VMGDMean, VMGDVariance = MeanAndVariance(len(Ns),K,VMGD)
        MINEMean, MINEVariance = MeanAndVariance(len(Ns),K,Mine)

###################################################################################
########################## Plots 4c in paper ######################################       
        fig = go.Figure([
        go.Scatter(
            x=Ns,
            y=SampleMIMean[:,0],
            line=dict(color='rgb(0,100,80)', width=3),
            mode='lines',
            name='True MI'
        ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((SampleMIMean[:,0]+SampleMIVariance[:,0]),(SampleMIMean[:,0]-SampleMIVariance[:,0])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(0,100,80,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        # ),
        go.Scatter(
            x=Ns,
            y=MMMIMean[:,0],
            line=dict(color='rgb(255,0,0)', width=3),
            mode='lines',
            name='I<sub>m+p</sub> MM'
        ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((MMMIMean[:,0]+MMMIVariance[:,0]),(MMMIMean[:,0]-MMMIVariance[:,0])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(255,0,0,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        # ),
        go.Scatter(
            x=Ns,
            y=GDMIMean[:,0],
            line=dict(color='rgb(255,0,0)', width=3, dash='dot'),
            mode='lines',
            name='I<sub>m+p</sub> GD'
        ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((GDMIMean[:,0]+GDMIVariance[:,0]),(GDMIMean[:,0]-GDMIVariance[:,0])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(255,0,0,0.5)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        # ),
            go.Scatter(
            x=Ns,
            y=BAMIMean[:,0],
            line=dict(color='rgb(0,0,255)', width=3),
            mode='lines',
            name='I<sub>post</sub> MM'
        ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((BAMIMean[:,0]+BAMIVariance[:,0]),(BAMIMean[:,0]-BAMIVariance[:,0])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(0,0,255,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        # ),
            go.Scatter(
            x=Ns,
            y=VMMIMean[:,0],
            line=dict(color='rgb(180,0,255)', width=3),
            mode='lines',
            name='I<sub>marg</sub> MM'
        ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((VMMIMean[:,0]+VMMIVariance[:,0]),(VMMIMean[:,0]-VMMIVariance[:,0])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(180,0,255,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        # ),
            go.Scatter(
            x=Ns,
            y=BAGDMean[:,0],
            line=dict(color='rgb(0,0,255)', width=3, dash='dot'),
            mode='lines',
            name='I<sub>post</sub> GD'
        ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((BAGDMean[:,0]+BAGDVariance[:,0]),(BAGDMean[:,0]-BAGDVariance[:,0])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(0,0,255,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        # ),
            go.Scatter(
            x=Ns,
            y=VMGDMean[:,0],
            line=dict(color='rgb(180,0,255)', width=3, dash='dot'),
            mode='lines',
            name='I<sub>marg</sub> GD'
        ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((VMGDMean[:,0]+VMGDVariance[:,0]),(VMGDMean[:,0]-VMGDVariance[:,0])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(180,0,255,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        # ),
        go.Scatter(
            x=Ns,
            y=MINEMean[:,0],
            line=dict(color='rgb(0,0,0)', width=3),
            mode='lines',
            name='MINE'
        # ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((MINEMean[:,0]+MINEVariance[:,0]),(MINEMean[:,0]-MINEVariance[:,0])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(0,0,0,0.2)',
        #     line=dict(color='rgb(0,0,0,0)'),
        #     mode="lines",
        #     hoverinfo="skip",
        #     showlegend=False
        )
        ])
        fig.update_xaxes(title_text="Number of Samples", type="log", dtick = "D2")
        fig.update_layout(
            xaxis={
                "tickmode": "array",
                "tickvals": pd.to_numeric(
                    [f"{n:.1g}" for n in np.array([500,1000,2000,5000])]
                ),
            }
        )
        fig.update_yaxes(title_text="MI Approximation")
        fig.update_layout(font=dict(size=30),showlegend=False)
        fig.update_layout(plot_bgcolor='white')
        fig.update_xaxes(
            # range = [400,5000],
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        # fig.write_image("LargeGMMConvergeNew.pdf")
        fig.show()
        
###################################################################################
########################## Plots 4a in paper ######################################  
        fig1 = go.Figure([
        # go.Scatter(
        #     x=Ns,
        #     y=SampleMIMean[:,1],
        #     line=dict(color='rgb(0,100,80)', width=3),
        #     mode='lines',
        #     name='True'
        # ),
        go.Scatter(
            x=Ns,
            y=MMMIMean[:,1],
            line=dict(color='rgb(255,0,0)', width=3),
            mode='lines',
            name='I<sub>m+p</sub> MM'
        ),
        go.Scatter(
            x=Ns,
            y=GDMIMean[:,1],
            line=dict(color='rgb(255,0,0)', width=3, dash='dot'),
            mode='lines',
            name='I<sub>m+p</sub> GD'
        ),
        go.Scatter(
            x=Ns,
            y=BAMIMean[:,1],
            line=dict(color='rgb(0,0,255)', width=3),
            mode='lines',
            name='I<sub>post</sub> MM'
        ),
        go.Scatter(
            x=Ns,
            y=BAGDMean[:,1],
            line=dict(color='rgb(0,0,255)', width=3, dash='dot'),
            mode='lines',
            name='I<sub>post</sub> GD'
        ),
        go.Scatter(
            x=Ns,
            y=VMMIMean[:,1],
            line=dict(color='rgb(180,0,255)', width=3),
            mode='lines',
            name='I<sub>marg</sub> MM'
        ),
        go.Scatter(
            x=Ns,
            y=VMGDMean[:,1],
            line=dict(color='rgb(180,0,255)', width=3, dash='dot'),
            mode='lines',
            name='I<sub>marg</sub> GD'
        ),
        go.Scatter(
            x=Ns,
            y=MINEMean[:,1],
            line=dict(color='rgb(0,0,0)', width=3),
            mode='lines',
            name='MINE'
        )
        ])
        fig1.update_xaxes(title_text="Number of Samples", type="log", dtick = "D2")
        fig1.update_layout(
            xaxis={
                "tickmode": "array",
                "tickvals": pd.to_numeric(
                    [f"{n:.1g}" for n in np.array([500,1000,2000,5000])]
                ),
            }
        )
        fig1.update_yaxes(title_text="Run Time", type="log", dtick = 1)
        fig1.update_layout(font=dict(size=25),showlegend=False)
        fig1.update_layout(plot_bgcolor='white')
        fig1.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig1.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        # fig1.write_image("LargeGMMTimeNew.pdf")
        fig1.show()
############################################################################### 