
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF
        

        # b. home production
        
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H= np.minimum(HM, HF)
        else: 
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        
            



        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    
    
    def solve_continuous(self, do_print=False):
        """ solve model using continuous optimization """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # define objective function
        obj = lambda x: -self.calc_utility(x[0],x[1],x[2],x[3])

        # define time constraints
        cons = [(0,24),(0,24),(0,24),(0,24)]

        # define initial guess
        guess = [12, 12, 12, 12]

        # run optimization
        res = optimize.minimize(obj, guess, bounds=cons)

        # extract optimal values
        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]

        # print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    

    def solve_wF_vec(self,discrete=False):
        #solve for female wages

        par = self.par
        sol = self.sol

        #a. loop over relative wage and solve model
        for i in par.wF_vec:
            par.wF = i
            if discrete: # for discrete choice model
                results = self.solve_discrete()
            else: #for continous choice model
                results = self.solve_continuous()
            # i. Find index 
            j = np.where(par.wF_vec ==i)[0][0]

            # ii. Store the results
            sol.LM_vec[j] = results.LM
            sol.HM_vec[j] = results.HM
            sol.LF_vec[j] = results.LF
            sol.HF_vec[j] = results.HF

        return sol
    
    def run_regression(self):
     # Run regression

        par = self.par
        sol = self.sol

        # Generating the relevant variables
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

        # Returns beta0 and beta1
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        return sol
    

    def estimate(self, alpha=None):
    # Define the objective function to minimize
        par = self.par
        sol = self.sol
        if alpha == None:
            def objective(y):
                par.sigma = y[1]
                par.alpha = y[0]
                self.solve_wF_vec()
                self.run_regression()
                return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2
            
            obj = lambda y: objective(y)
            guess = [0.5, 0.5]  # initial guess for sigma and alpha
            bounds = [(1e-5)*2]  # bounds for sigma and alpha
            result = optimize.minimize(objective, guess, method='Nelder-Mead', bounds=bounds)
            print(f'sigma = {result.x[1]:.2f}, alpha = {result.x[0]:.2f} -> beta0 = {model.sol.beta0:.2f}, beta1 = {model.sol.beta1:.2f}')
        else:
            # i. objective function (to minimize)
            def objective(y):
                par.alpha = alpha 
                par.sigma = y[0] 
                self.solve_wF_vec()
                self.run_regression()
                return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

            obj = lambda y: objective(y)
            guess = [0.5]
            bounds =  [(1e-5)]

            # ii. optimizer
            result = optimize.minimize(obj,
                                guess,
                                method='Nelder-Mead',
                                bounds=bounds)
          
            return result