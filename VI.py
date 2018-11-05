"""
Ted Moskovitz, 2018
Simple Implementation of Variational Inference

A simple model for variational inference with data of the form (x,y),
where y ~ R and x ~ R^d. The regression model is as follows:
y ~ Normal(x^T w, lambda^-1), w ~ Normal(0, diag(alpha_1,..., alpha_d)^-1),
alpha_k ~ Gamma(a0, b0), lambda ~ Gamma(e0, f0)

The VI algorithm will approximate the posterior distribution:
p(w, alpha_1,...,alpha_d, lambda |y,x)
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import digamma, gamma, gammaln
plt.rcParams['figure.figsize'] = (12.0, 9) # set default size of plots
plt.rc('text', usetex=True)
import seaborn as sns; sns.set()

class VI:
    
    def __init__(self, data):
        """
        initialize variational inference alg.
        args:
          data: dictionary with 'X', 'y', and 'z' keys
        """
        self.X = data['X'].T # d x N
        self.y = data['y']
        self.z = data['z']
        self.d = self.X.shape[0]
        self.N = self.X.shape[1]
        self.a0 = 1e-16
        self.b0 = 1e-16
        self.e0 = 1
        self.f0 = 1
        
        # initialize variational parameters
        self.mu = np.random.normal(loc=0.0, scale=0.01, size=self.d)
        self.sigma = np.zeros([self.d, self.d])
        self.a1 = self.a0
        self.b1s = np.zeros(self.d)
        self.e1 = self.e0
        self.f1 = self.f0
    
    def ELBO(self):
        """
        calculate the variational objective function
        """
        a1 = self.a1
        b1s = self.b1s
        a0 = self.a0
        b0 = self.b0
        e1 = self.e1
        f1 = self.f1
        e0 = self.e0
        f0 = self.f0
        mu = self.mu
        sigma = self.sigma
        N = self.N
        d = self.d
        y = self.y
        X = self.X
        eps = 1e-7
        
        L = 0.0
        
        # E_q(w,alpha_1:d)[ln p(w|alpha_1:d)]
        tmp = sigma + np.dot(mu.reshape(-1,1), mu.reshape(-1,1).T)
        for k in range(d):
            L += 0.5 * (digamma(a1) - np.log(b1s[k] + eps) - (a1 / (b1s[k] + eps))                         * tmp[k,k])
            
        
        # E_q(lambda)[ln p(lambda)]
        L += (e0 - 1) * (digamma(e1) - np.log(f1)) - f0 * (e1 / f1)
        
        
        # sum_k E_q(alpha_k)[ln p(alpha_k)]
        L += np.sum((a0 - 1) * (digamma(a1) - np.log(b1s + eps)) - b0 * (a1 / (b1s + eps)))
        
        # sum_i E_q(w,lambda) [ln p(y_i | x_i, w, lambda)]
        L += (N / 2.0) * (digamma(e1) - np.log(f1 + eps))  
        for i in range(N):
            L -= 0.5 * (e1 / f1) * ((y[i] - np.dot(X[:,i], mu))**2 + np.dot(np.dot(X[:,i], sigma), X[:,i]))
            
        # H[q(w)]
        sign, logdet = np.linalg.slogdet(sigma)
        L += 0.5 * sign * logdet
        
        # H[q(lambda)]
        L += e1 - np.log(f1) + gammaln(e1) + (1.0 - e1) * digamma(e1)
        
        # sum_k H[q(alpha_k)]
        L += np.sum(a1 - np.log(b1s) + gammaln(a1) + (1.0 - a1) * digamma(a1))
        
        return L
        
        
    def run(self, T):
        """
        run VI for T iterations
        args:
          T: number of iterations
        returns:
          objective function history (list)
        """
        L_hist = []
        for t in range(1, T+1):
            # update q(alpha_k)
            self.a1 = self.a0 + 0.5
            smu = self.mu.reshape(-1,1)
            self.b1s = self.b0 + 0.5 * np.diag(np.dot(smu, smu.T) + self.sigma).reshape(-1,)
            
            # update q(lambda)
            self.e1 = self.e0 + 0.5 * self.N
            self.f1 = self.f0
            for i in range(self.N):
                self.f1 += 0.5 * ((self.y[i] - np.dot(self.X[:,i], self.mu))**2 +                                   np.dot(np.dot(self.X[:,i], self.sigma), self.X[:,i]))
            
            # update q(w) 
            self.sigma =  np.linalg.inv(np.diag(self.a1 / self.b1s)                 + (self.e1 / (self.f1 + 1e-5)) * np.dot(self.X, self.X.T))
            self.mu = np.dot(self.sigma * (self.e1 / (self.f1 + 1e-5)), np.dot(self.X, self.y)).reshape(-1,)
            
            # evaluate objective function
            L_t = self.ELBO()
            if t % 25 == 0: 
                print ('Iteration {}/{}, objective value = {}'.format(t, T, L_t))
            L_hist.append(L_t)
        
        return L_hist



def main(data_id, n_iters):
    data_names = ['X', 'y', 'z']
    data = {}
    for dt in data_names:
        data[dt] = np.genfromtxt('data_csv/{}_set{}.csv'.format(dt, data_id), delimiter=',')

    # run algorithm
    vi = VI(data)
    L_hist = vi.run(n_iters)

    # plot objective function
    ts = np.arange(len(L_hist)) * (n_iters / len(L_hist))
    plt.plot(ts, L_hist, lw=4)
    plt.xlabel('$t$', fontsize=28)
    plt.ylabel('$\mathcal{L}_{t}$', fontsize=28)
    plt.tick_params(labelsize=23)
    plt.title('Objective Function, Dataset {}'.format(data_id), fontsize=30)
    plt.show()

    # print 1 / E[alpha_k] as a function of k
    plt.plot(vi.b1s / vi.a1)
    plt.tick_params(labelsize=20)
    plt.xlabel('$k$', fontsize=26)
    plt.ylabel('1/E[alpha-k]', fontsize=26)
    plt.title('Inverse Expected alpha-k, Dataset {}'.format(data_id), fontsize=28)
    plt.show()

    # 1 / E[lambda]
    print (vi.f1 / vi.e1)

    # plot fit
    y_hat = np.dot(data['X'], vi.mu)
    plt.plot(data['z'], y_hat, label='$\hat{y}$', lw=4)
    plt.plot(data['z'], 10*np.sinc(data['z']), label='$10sinc(z)$', lw=4)
    plt.scatter(data['z'], data['y'], label='$y$', color='C2')
    plt.legend(fontsize=24)
    plt.tick_params(labelsize=20)
    plt.xlabel('$z$', fontsize=26)
    plt.ylabel('$y$', fontsize=26)
    plt.title('Dataset {}'.format(data_id), fontsize=28)
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', nargs='?', const=1, type=int)
    parser.add_argument('--n_iters', nargs='?', const=500, type=int)
    args = parser.parse_args()
    main(args.data_id, args.n_iters)

