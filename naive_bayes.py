"""
Ted Moskovitz, 2018
Simple Naïve Bayes Model for Binary Classification
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import factorial
from scipy.misc import comb as choose
import sys
import math
plt.rcParams['figure.figsize'] = (15.0, 5.0) # set default size of plots

class naive_bayes:
    
    def __init__(self, Xtr, ytr):
        '''
        initialize classifier with training data
        args:
          Xtr: input features (n x d)
          ytr: associated class labels (n x 1)
        '''
        self.Xtr = Xtr
        self.ytr = ytr
        self.n_tr = Xtr.shape[0]
        self.num_features = Xtr.shape[1]
        self.num_classes= len(np.unique(ytr))

    def negbin(self, k, r, p):
        '''
        sample from negative binomial distribution
        '''
        a = choose(k+r-1, k)
        b = p**k
        c = (1.0 - p)**r
        return a * b * c
        
    def get_label_priors(self, e=1.0, f=1.0):
        '''
        return label priors p(y_test = 1 | y_train)
        args:
          e, f: distribution parameters
        returns:
          prior probabilities
        '''
        n0_tr = self.n_tr - np.sum(self.ytr)
        n1_tr = np.sum(self.ytr)

        py0 = (f + n0_tr) / (self.n_tr + e + f)
        py1 = (e + n1_tr) / (self.n_tr + e + f)
        return np.asarray([py0, py1])
    
    def classify(self, X, a=1.0, b=1.0):
        '''
        data likelihood under assumed negative binomial distribution
        args:
          X: input data (n x d)
          a, b: distribution parameters
        returns:
          preds: model predictions
          label_probs: model probability of each prediction
        '''
        n = X.shape[0]
        
        Py = self.get_label_priors()
        label_probs = np.zeros([n, self.num_classes])
        preds = np.zeros([n,])
        for i in range(n):
            for c in range(self.num_classes):
                Xc = self.Xtr[self.ytr == c]
                nc = Xc.shape[0]
                Pxy = self.negbin(X[i,:], a + np.sum(Xc, axis=0), 1.0 / (nc + b + 1.0))
                label_probs[i, c] = np.prod(Pxy) * Py[c]
            label_probs[i] /= np.sum(label_probs[i])
            preds[i] = np.argmax(label_probs[i])
            
        return preds, label_probs 
    
    def get_acc(self, y, yhat):
        '''
        return accuracy of model predictions
        args:
          y: ground truth labels
          yhat: predicted labels
        returns:
          accuracy
        '''
        return np.sum((y == yhat).astype(int)) / float(len(y))


def show_feature_vals(email_idxs, name=None):
    '''
    display features for given emails
    '''
    for i,idx in enumerate(email_idxs):
        plt.plot(X_test[idx,:],  label=r'email features')
        plt.plot(E0, label='E[lambda_0]')
        plt.plot(E1, label='E[lambda_1]')
        plt.xticks(np.arange(len(E0)), feature_labels, rotation=70)
        plt.ylabel('feature value', fontsize=20)
        plt.xlabel('feature', fontsize=20)
        pred_probs = probs[idx]
        plt.title('email {}: ground truth label = {}, prediction probs = {}'.format(idx+1,
                                                                                    y_test[idx], probs[idx]),
                  fontsize=22)
        plt.legend(fontsize=16)
        if name: plt.savefig('fig_{}{}.png'.format(name,i+1));
        plt.show()

def confusion_matrix(y, yhat):
    '''
    generate a confusion matrix based on
    ground truth labels (y) and predictions (yhat)
    '''
    y_0idxs = set(np.where(y == 0.0)[0])
    y_1idxs = set(np.where(y == 1.0)[0])
    yh_0idxs = set(np.where(yhat == 0.0)[0])
    yh_1idxs = set(np.where(yhat == 1.0)[0])
    
    true_positives = y_1idxs.intersection(yh_1idxs)
    n_tp = len(list(true_positives))
    true_negatives = y_0idxs.intersection(yh_0idxs)
    n_tn = len(list(true_negatives))
    false_positives =  y_0idxs.intersection(yh_1idxs)
    n_fp = len(list(false_positives))
    false_negatives = y_1idxs.intersection(yh_0idxs)
    n_fn = len(list(false_negatives))
    
    cm = np.zeros([2,2])
    cm[0,0] = n_tn
    cm[0,1] = n_fn
    cm[1,0] = n_fp
    cm[1,1] = n_tp
    return cm


def main():
    # import data
    X_train = np.genfromtxt('data_csv/X_train.csv', delimiter=',')
    y_train = np.genfromtxt('data_csv/label_train.csv', delimiter=',')
    n_train = y_train.shape[0]
    X_test = np.genfromtxt('data_csv/X_test.csv', delimiter=',')
    y_test = np.genfromtxt('data_csv/label_test.csv', delimiter=',')
    n_test = y_test.shape[0]
    dim = X_train.shape[1]

    # test classifier, produce confusion matrix
    nb = naive_bayes(X_train, y_train)
    preds, probs = nb.classify(X_test)
    print ("accuracy: ", nb.get_acc(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    print ('Confusion Matrix:')
    print (cm)
    print ('Format is:')
    print ('             ground truth')
    print ('             ------------')
    print ('   predicted|')


    # analysis of mislabeled emails
    incorrect_idxs = np.where(y_test !=  preds)[0]
    email_idxs = np.random.choice(incorrect_idxs, size=3, replace=False)

    feature_labels = ['make', 'address', 'all', '3d', 'our',
              'over', 'remove', 'internet', 'order', 'mail',
              'receive', 'will', 'people', 'report', 'addresses',
              'free', 'business', 'email', 'you', 'credit',
              'your', 'font', '000', 'money', 'hp',
              'hpl', 'george', '650', 'lab', 'labs',
              'telnet', '857', 'data', '415', '85',
              'technology', '1999', 'parts', 'pm', 'direct',
              'cs', 'meeting', 'original', 'project', 're',
              'edu', 'table', 'conference', ';', '(',
              '[', '!', '$', '#']

    # E[lambda] when lambda ~ Gamma(a,b) is a/b
    # In this case, the distribution is Gamma(a + sum_i x_i, b + N)
    a, b = 1.0, 1.0

    # E[lambda_0]
    a0 = a + np.sum(X_train[y_train == 0.0], axis=0)
    b0 = b + len(y_train[y_train == 0.0])
    E0 = a0 / b0

    # E[lambda_1]
    a1 = a + np.sum(X_train[y_train == 1.0], axis=0)
    b1 = b + len(y_train[y_train == 1.0])
    E1 = a1 / b1

    show_feature_vals(email_idxs, name='c')

    # find most uncertain emails
    probs0 = probs[:,0]
    dists = np.abs(probs0 - 0.5)
    sorted_dists = np.sort(dists)
    sorted_dist_idxs = np.argsort(dists)
    # 3 closest
    most_ambig_idxs = sorted_dist_idxs[:3]

    show_feature_vals(most_ambig_idxs, name='d')

if __name__=='__main__':
    main()




