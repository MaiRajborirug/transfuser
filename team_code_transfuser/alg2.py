import numpy as np
from scipy import optimize
import math
import time
from multiprocessing import Pool

class Algorithm2:
    def __init__(self) -> None:
        self.a_bounds = (-13.0, 11.0)
        self.phi_bounds = (-3.0, 3.0)
        self.a_number_points = 20
        self.phi_number_points = 20
        self.cost_func_weight = 1000

    def optimize_discrete(self, args):
        def cost_func(u_t, *args):
            # u_t: current control action 
            # u_t_nom (*args): a singular control action used to compute cost 
            # certify: function that takes in control action and returns
            # whether it is certified
            # non-certified funcs are max possible float
            # returns: cost of u_t
            [a, phi] = u_t.tolist()
            (a_nom, phi_nom, certify) = args
            # return (5*(a_nom - a)**2 + (phi_nom - phi)**2) + self.cost_func_weight * (1 - certify(u_t))
            return ((a_nom - a)**2 + 20*(phi_nom - phi)**2) + self.cost_func_weight * (1 - certify(u_t))

        before_op = time.time()
        (a_nom, phi_nom, certify) = args
        
        self.res = optimize.brute(func=cost_func,
                             ranges=((self.a_bounds[0] + a_nom, self.a_bounds[1] + a_nom), (self.phi_bounds[0] + phi_nom, self.phi_bounds[1] + phi_nom)),
                             args=args,
                             Ns=3,
                             finish=None,
                             full_output=True)

        aft_op = time.time()
        # print("alg2 time", aft_op - before_op)
        
        [x, fval, grid, _] = self.res
        return x, fval

            
    def run(self, u_t_nom, certify):
        # u_t_nom: a singular control action assumed to be available at all times (not based on control objective)
        # returns: nothing if there is no certified control action or u_t, the desired control action at time t
        b = 0
        a_nom, phi_nom = u_t_nom
        args = (a_nom, phi_nom, certify)
        optimized_u, cost = self.optimize_discrete(args)

        if (cost < np.finfo(np.float32).max):
            return optimized_u
        else:
            return None
        
        
        
                    
                    
                    
