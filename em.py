"""
Ted Moskovitz, Fall 2018
A Simple Implementation of the EM Algorithm

This script applies EM to find a low-rank approximation
of movie rating data. This data takes the form of a sparse
ratings matrix R, and will be parameterized by a matrix U where each
row represents a user, and a matrix V representing movies. We assume
only a subset Omega of all ratings in R are known; the goal is to
learn U and V such that we can accurately predict the remaining
values in R. The EM algorithm attempts to maximize the join log-
likelihood log p(R,U,V).

The model variables are defined as follows:
- u_i \in R^d, v_j \in R^d for i = 1,...,N and j = 1,...,M
- R \in R^NxM, R_ij is +/- 1
- R_ij | U,V ~ Bernoulli(CDF(u_i^T v_j / sigma)) for all (i,j) \in Omega,
  where CDF() is the CDF of a standard normal random variable
- u_i ~ Normal(0, cI)
- v_j ~ Normal(0, cI)
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import math

def CDF(x): return norm.cdf(x);
def PDF(x): return norm.pdf(x);

class EM:
    
    def __init__(self, Rtr, d=5, var=1.0, c=1.0): 
        '''
        set params
        args:
          Rtr  training data, n_ratings x 3
          d: dimension of parameter vectors
          var: variance parameter
          c: user vector variance parameter
        '''
        Rtr = Rtr.astype(np.int32)
        self.Rtr = Rtr
        self.Omega_sz = np.int32(Rtr.shape[0])
        self.d = np.int32(d)
        self.var = np.float32(var)
        self.c = np.float32(c)
        self.loglis = []
        
        self.N = np.int32(np.max(R_tr[:,0]))
        self.M = np.int32(np.max(R_tr[:,1]))
        
        self.R = np.zeros(self.N * self.M).reshape(self.N, self.M).astype(np.int32)
        for i in range(self.Omega_sz):
            self.R[Rtr[i,0]-1, Rtr[i,1]-1] = Rtr[i,2]
            
        self.Rpos = (self.R > 0).astype(float)
        self.Rneg = (self.R < 0).astype(float)
        
        self.U = np.random.normal(loc=0.0, scale=0.1, size=(self.N, self.d)).astype(np.float32)
        self.V = np.random.normal(loc=0.0, scale=0.1, size=(self.M, self.d)).astype(np.float32)
        self.E = np.zeros(self.N * self.M).reshape(self.N, self.M).astype(np.float32)
        
    def E_step(self):
        '''
        compute E_{q}[\phi]
        '''
        sigma = np.sqrt(self.var)
        # instead of u^T v do UV^T based on shapes
        sims = np.dot(self.U, self.V.T)
        inner = (-1.0 * sims) / sigma
        
        Epos = sims + sigma * (PDF(inner) / (1.0 - CDF(inner) + 1e-6))
        Eneg = sims - sigma * (PDF(inner) / (CDF(inner) + 1e-6))
        self.E = self.Rpos * Epos + self.Rneg * Eneg
          
    def M_step(self):
        '''
        maximization step
        '''
        sigma = np.sqrt(self.var)
        oldU = self.U
        self.U = np.dot(inv((1.0/self.c)*np.eye(self.d) + np.dot(self.V.T, self.V)/sigma),
                        np.dot(self.V.T, self.E.T)/sigma).T
        
        self.V = np.dot(inv((1.0/self.c)*np.eye(self.d) + np.dot(oldU.T, oldU)/sigma),
                        np.dot(oldU.T, self.E)/sigma).T
        
    def logli(self):
        '''
        compute the log likelihood of the model
        '''
        logli = 0.0
        sigma = np.sqrt(self.var)
        const = np.log(1.0 / (2 * math.pi * self.c))
        U = self.U
        V = self.V
        c = self.c
        R_01 = self.R.copy()
        R_01[R_01 < 0] = 0
        CDF_probs = CDF(np.dot(U,V.T)/sigma)
        logli = np.sum(R_01 * np.log(CDF_probs + 1e-6) + (1.0-R_01) * np.log(1.0 - CDF_probs + 1e-6) + const)
        logli += np.sum((1.0/(2*c)) * np.dot(U,U.T)) + np.sum((1.0/(2*c)) * np.dot(V,V.T)) 
        return logli    
    
    def predict(self, R_te, n_classes=2):
        '''
        compute test log likelihoods
        '''
        R_te = R_te.astype(int)
        N = R_te.shape[0]
        sigma = np.sqrt(self.var)
        const = np.log(1.0 / (2 * math.pi * self.c))
        preds = np.zeros(N)
        classes = [-1.0, 1.0]
        
        for idx in range(N):
            i = R_te[idx,0] - 1
            j = R_te[idx,1] - 1
            u_i = self.U[i,:].reshape(-1,)
            v_j = self.V[j,:].reshape(-1,)
            sim = np.dot(u_i, v_j)
            p = CDF(sim / sigma)
            class_probs = np.array([1.0-p, p])
            preds[idx] = np.argmax(class_probs)
    
        preds[preds == 0] = -1
        return  preds
    
    def get_acc(self, y, y_hat):
        '''
        compute the accuracy of the model predictions
        args:
          y: ground truth labels
          y_hat: predicted labels
        returns:
          accuracy
        '''
        return sum(y == y_hat) / float(len(y_hat))
        
    def run(self, T, R_te=None):
        '''
        run algorithm for T steps 
        '''
        self.loglis = []
        start = time.time()
        for t in range(1,T+1):
            self.E_step()
            self.M_step()
            
            ll = self.logli() #self.U, self.V, self.Rtr, self.var, self.c
            self.loglis.append(ll)
            
            if t % 5 == 0:
                y_hat = self.predict(R_te)
                acc = self.get_acc(R_te[:,2], y_hat)
                print ('iteration {}/{}: log-likelihood = {}, test accuracy = {}, time = {} min'.format(t,
                                                                    T,ll,acc,
                                                                    (time.time()-start)/60.))
            else:
                print ('iteration {}/{}: log-likelihood = {}, time = {} min'.format(t,
                                                                    T,ll,
                                                                    (time.time()-start)/60.))
                

                
        end = time.time()
        print ('Total time: {} min'.format((end-start)/60.0))

def confusion_matrix(y, yhat):
    '''
    generate a confusion matrix
    args:
      y: ground truth labels
      yhat: predicted labels
    returns:
      2x2 confusion matrix
    '''
    y_n1idxs = set(np.where(y == -1.0)[0])
    y_1idxs = set(np.where(y == 1.0)[0])
    yh_n1idxs = set(np.where(yhat == -1.0)[0])
    yh_1idxs = set(np.where(yhat == 1.0)[0])
    
    true_positives = y_1idxs.intersection(yh_1idxs)
    n_tp = len(list(true_positives))
    true_negatives = y_n1idxs.intersection(yh_n1idxs)
    n_tn = len(list(true_negatives))
    false_positives =  y_n1idxs.intersection(yh_1idxs)
    n_fp = len(list(false_positives))
    false_negatives = y_1idxs.intersection(yh_n1idxs)
    n_fn = len(list(false_negatives))
    
    cm = np.zeros([2,2])
    cm[0,0] = n_tn
    cm[0,1] = n_fn
    cm[1,0] = n_fp
    cm[1,1] = n_tp
    return cm

def main():
    # run EM algorithm
    R_tr = np.genfromtxt('data_csv/ratings.csv', delimiter=',')
    R_te = np.genfromtxt('data_csv/ratings_test.csv', delimiter=',')
    movie_names = []
    with open('movies.txt') as f:
        movie_names = f.readlines()
     
    em = EM(R_tr, d=5, var=1.0, c=1.0)

    em.run(100, R_te=R_te)

    plt.plot(np.arange(len(em.loglis[2:]))+2, em.loglis[2:])
    plt.xlabel('Iteration', fontsize=24)
    plt.ylabel('Log-Likelihood', fontsize=24)
    plt.title('EM Algorithm Performance', fontsize=26)
    plt.tick_params(labelsize=20)
    plt.show()

    # run different initializations
    perfs = []
    for i in range(5):
        print ('\nRun {}'.format(i+1))
        em = EM(R_tr, d=5, var=1.0, c=1.0)
        em.run(100, R_te=R_te)
        perfs.append(em.loglis)

    for i in range(5):
        loglis = perfs[i]
        plt.plot(np.arange(len(loglis[19:]))+20, loglis[19:], label='run {}'.format(i+1), lw=3)
        plt.xlabel('Iteration', fontsize=24)
        plt.ylabel('Log-Likelihood', fontsize=24)
        plt.title('EM Algorithm Performance', fontsize=26)
        plt.legend(fontsize=20)
        plt.tick_params(labelsize=20)
    plt.show()


    # predict test ratings
    y_hat = em.predict(R_te)
    cm = confusion_matrix(R_te[:,2], y_hat)
    print ('Confusion Matrix:')
    print (cm)
    print ('Format is:')
    print ('             ground truth')
    print ('             ------------')
    print ('   predicted|')

if __name__=="__main__":
    main() 

