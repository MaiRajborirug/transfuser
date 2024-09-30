import numpy as np
from scipy import optimize
import math
import time
from multiprocessing import Pool

class Algorithm2:
    def __init__(self) -> None:
        self.a_bounds = (-13.0, 11.0)
        self.alpha_bounds = (-5.0, 5.0)
        self.a_number_points = 20
        self.alpha_number_points = 20
        self.cost_func_weight = 1000

    def optimize_discrete(self, args):
        def cost_func(u_t, *args):
            # u_t: current control action 
            # u_t_nom (*args): a singular control action used to compute cost 
            # certify: function that takes in control action and returns
            # whether it is certified
            # non-certified funcs are max possible float
            # returns: cost of u_t
            [a, alpha] = u_t.tolist()
            (a_nom, alpha_nom, certify) = args
            
            # max: 1 - certify(u_t) = 1
            # but about < 300 pixel determine the cost out ff 460800 pixel -> use
            # max: (a_nom - a)**2/169.0 + 20*(alpha_nom - alpha)**2/25.0 =1
            return (460800/300)*(1 - certify(u_t))  + 0.1 * ((a_nom - a)**2/169.0 + 20*(alpha_nom - alpha)**2/25.0) 

        before_op = time.time()
        (a_nom, alpha_nom, certify) = args
        
        self.res = optimize.brute(func=cost_func,
                             ranges=((self.a_bounds[0] + a_nom, self.a_bounds[1] + a_nom), (self.alpha_bounds[0] + alpha_nom, self.alpha_bounds[1] + alpha_nom)),
                             args=args,
                             Ns=3,
                             finish=None,
                             full_output=True)

        
        [x, fval, grid, _] = self.res
        # breakpoint()
        return x, fval

            
    def run(self, u_t_nom, certify):
        # u_t_nom: a singular control action assumed to be available at all times (not based on control objective)
        # returns: nothing if there is no certified control action or u_t, the desired control action at time t
        b = 0
        a_nom, alpha_nom = u_t_nom
        args = (a_nom, alpha_nom, certify)
        optimized_u, cost = self.optimize_discrete(args)

        if (cost < np.finfo(np.float32).max):
            return optimized_u
        else:
            return None
        
        
        
                    
                    
                    
