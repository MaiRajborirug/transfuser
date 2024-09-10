import numpy as np
from scipy import optimize
import numba
from numba import njit, prange
import multiprocessing
import numexpr as ne
from cuda import cuda, nvrtc
import time

# CUDA error checking function
def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

class algorithm_1:
    def __init__(self) -> None:
        self.v_bounds = [1, 100]
        self.D_bounds = [1, 100]
        self.psi_i_bounds = [1, 100]
        self.a_i_bounds = [0.1, 1000000]

        # self.candidates = {
        #     0: [],
        #     1: self.v_bounds,
        #     "D": self.D_bounds,
        #     2: self.psi_i_bounds,3
        #     "a_i": self.a_i_bounds
        # }

        self.candidates = [
                [],
                self.v_bounds,
                self.psi_i_bounds,
                self.D_bounds
        ]

        self.theta_guess_from_59 = None

        # # Create program
        # err, prog = nvrtc.nvrtcCreateProgram(str.encode(eqn_54), b"eqn_54.cu", 0, [], [])

        # # Compile program
        # opts = [b"--fmad=false", b"--gpu-architecture=compute_75"]
        # err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)

        # # Get PTX from compilation
        # err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
        # ptx = b" " * ptxSize
        # err, = nvrtc.nvrtcGetPTX(prog, ptx)
        
        # # Initialize CUDA Driver API
        # err, = cuda.cuInit(0)

        # # Retrieve handle for device 0
        # err, cuDevice = cuda.cuDeviceGet(0)

        # # Create context
        # err, context = cuda.cuCtxCreate(0, cuDevice)

        # # Load PTX as module data and retrieve function
        # ptx = np.char.array(ptx)
        # # Note: Incompatible --gpu-architecture would be detected here
        # err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
        # ASSERT_DRV(err)
        # err, kernel = cuda.cuModuleGetFunction(module, b"eqn_54")
        # ASSERT_DRV(err)

        # NUM_THREADS = 512  # Threads per block
        # NUM_BLOCKS = 32768  # Blocks per grid

    def solve_eqn_60_for_0(self, f, X_i_t):
        theta = np.arctan(X_i_t / f)
        theta_candidates = [theta, (theta + np.pi) % (2 * np.pi)]
        self.candidates[0] += theta_candidates

    def solve_theta_guess_from_59(self, f, X_i_t):
        theta = np.arctan(-f / X_i_t)
        self.theta_guess_from_59 = [theta, (theta + np.pi) % (2 * np.pi)]
        self.candidates[0] += self.theta_guess_from_59

    def solve_eqn_59_for_0(self, f, X_i_t):
        if (self.theta_guess_from_59 is None):
            self.solve_theta_guess_from_59(f, X_i_t)
        self.candidates[1].append(0)
        
    
    def solve_eqn_57_for_0(self, f, X_i_t, psi_e):
        if (self.theta_guess_from_59 is None):
            self.solve_theta_guess_from_59(f, X_i_t)
        self.candidates[2].append(psi_e * 2)

    
    def eqn_54(self, x, *args):
        [a_i] = x
        (f, mu_i, nu_i, X_i_t, Y_i_t, v_i, v_e, psi_i, psi_e, a_e, theta, phi_e, CHI, D) = args
        # TODO: speed this thing up 
        # look into https://github.com/pydata/numexpr
        # look into https://numba.pydata.org/
        # all terms can theoretically be computed parallel
        
        term1 = phi_e * f
        term2 = 2 * mu_i * nu_i / Y_i_t
        term3 = (X_i_t ** 2) * phi_e / f
        D_CHI = D * CHI
        term4 = X_i_t * a_e
        term5 = a_i * f * np.sin(theta) 
        term6 = -f * psi_e * v_e
        COS_THETA = np.cos(theta)
        term7 = -X_i_t * a_i * COS_THETA
        term8 = 2 * f * psi_e * v_i * COS_THETA
        term9 = -f * psi_i * v_i * COS_THETA    
        term10 = np.sin(theta) * 2 * X_i_t * psi_e * v_i - X_i_t * psi_i * v_i

        result = term1 + term2 + term3 + (term4 + term5 + term6 + term7 + term8 + term9 + term10) / D_CHI
        return result
    
    def negate_function(self, func):
        def neg_func(args):
            return func(args) * -1
        return neg_func

    def solve_grid_search(self, args):
        (eqn_54_args, func) = args
        # grid search over a and D
        res = optimize.brute(
            func=func,
            ranges=(tuple(self.a_i_bounds),),
            args=eqn_54_args, 
            Ns=100,
            full_output=True,
            finish=None)
        [x, y, _, _] = res

        return (x, y)

    def optimize(self, f, mu_i, nu_i, X_i_t, Y_i_t, v_e, psi_e, a_e, phi_e, CHI, findmin=True):
        # TODO: vectorize the candidate list
        self.solve_eqn_60_for_0(f, X_i_t)
        self.solve_eqn_59_for_0(f, X_i_t)
        self.solve_eqn_57_for_0(f, X_i_t, psi_e)
        
        # at this point, there should be candidate lists for 
        # v, psi_e, and theta

        # for each possible permutation of the candidate lists,
        # do grid search for D and a given the permutation
        # TODO: do these grid searches in parallel somehow
        func=(self.eqn_54 if findmin else self.negate_function(self.eqn_54))

        args = []

        # create a 2d array,
        # where each row is one
        # permutation of candidates 
        for theta in self.candidates[0]:
            for v_i in self.candidates[1]:
                for psi_i in self.candidates[2]:
                    for D in self.candidates[3]:
                        eqn_54_args = (f, mu_i, nu_i, X_i_t, Y_i_t, 
                                    v_i, v_e, psi_i, psi_e, a_e,
                                    theta, phi_e, CHI, D)
                        args.append((eqn_54_args, func))

        xs = []
        ys = []
        
        with multiprocessing.Pool(len(args)) as pool:
            results = pool.map(self.solve_grid_search, args)
            for i in range(len(results)):
                result = results[i]
                (x, y) = result
                # map results preserve order of args
                xs.append((x,) + args[i][0])
                ys.append(y)

        y_idx = np.argmin(ys)
        return xs[y_idx]       

                    
                    

