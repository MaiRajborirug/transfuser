import ctypes
import math
import numpy as np
import pycuda.driver as cuda
import pycuda.autoprimaryctx  # when there is torch.cuda.set_device('cuda:0')
# import pycuda.autoinit # issue with cuda runtime API library : https://github.com/lebedov/scikit-cuda/issues/330

from pycuda.compiler import SourceModule
import pycuda
from pathlib import Path
import time
import os
import cv2 as cv

from memory_profiler import profile

pycuda.tools.get_default_device(0)  # initialize cuda device
# project_root = '/home/haoming/git/transfuser'


"""
:param f: focal length
:param camera_intrinsics: (camera_width, camera_height)
:param X, Y: The X and Y value of each pixel
:param certification_offset: the original offset used to certify the pixels
"""
class Algorithm1:
    def __init__(self, f, camera_intrinsics, X, Y, certification_offset, cuda_path):
        # cuda_source_code = Path(project_root + '/team_code_transfuser/alg1_pycuda_copy.cu').read_text()
        cuda_source_code = Path(cuda_path).read_text()

        # cuda_source_code = Path('alg1_pycuda_copy.cu').read_text()
        kernel = SourceModule(cuda_source_code)
        
        
        # unpack vehicle constants
        self.f = np.float32(f)
        self.camera_width, self.camera_height = camera_intrinsics
        self.X, self.Y = np.float32(X), np.float32(Y)
        self.certification_offset = np.float32(certification_offset)

        # define number of steps when doing grid search best object angle theta
        self.gridsearchsize = 50 # 

        # number of pixels
        self.N_pixels = X.size

        # self.THREAD_PER_BLOCK = 32 * 16
        self.block_dim = (20, 25, 1)

        # self.NUM_BLOCKS = 2 ** math.ceil(math.log2((self.N_pixels // self.THREAD_PER_BLOCK) + 1))
        self.grid_dim = (30, 32)

        # setup cuda kernel
        self.certify_u_for_mu = kernel.get_function("certify_u_for_mu")
        self.certify_u_for_nu = kernel.get_function("certify_u_for_nu")

        # buffer size calculations
        self.fp32_vector_buffersize = self.N_pixels * 4
        self.bool_vector_buffersize = self.N_pixels

        # allocate memory for cuda
        self.d_mus = cuda.mem_alloc(self.fp32_vector_buffersize)
        self.d_nus = cuda.mem_alloc(self.fp32_vector_buffersize)
        self.d_Xs = cuda.mem_alloc(self.fp32_vector_buffersize)
        self.d_Ys = cuda.mem_alloc(self.fp32_vector_buffersize)
        self.d_offsets = cuda.mem_alloc(self.fp32_vector_buffersize)
        self.d_animateds = cuda.mem_alloc(self.bool_vector_buffersize)
        self.d_depth_upper_bound = cuda.mem_alloc(self.fp32_vector_buffersize)
        self.d_depth_lower_bound = cuda.mem_alloc(self.fp32_vector_buffersize)

        # result buffer
        self.d_mu_is_certifieds = cuda.mem_alloc(self.bool_vector_buffersize)
        self.d_nu_is_certifieds = cuda.mem_alloc(self.bool_vector_buffersize)
        self.d_mu_b_outs = cuda.mem_alloc(self.fp32_vector_buffersize)
        self.d_nu_b_outs = cuda.mem_alloc(self.fp32_vector_buffersize)
        self.d_mu_i_outs = cuda.mem_alloc(self.fp32_vector_buffersize)
        self.d_nu_i_outs = cuda.mem_alloc(self.fp32_vector_buffersize)
        self.d_mu_dot_outs = cuda.mem_alloc(self.fp32_vector_buffersize)
        self.d_nu_dot_outs = cuda.mem_alloc(self.fp32_vector_buffersize)

        # copy inputs into cuda device
        cuda.memcpy_htod(self.d_Xs, self.X)
        cuda.memcpy_htod(self.d_Ys, self.Y)
        cuda.memcpy_htod(self.d_offsets, self.certification_offset)

        # create result numpy arrays
        self.mu_is_certifieds = np.empty((self.N_pixels, ), dtype=np.bool)
        self.nu_is_certifieds = np.empty((self.N_pixels, ), dtype=np.bool)
        self.mu_b_outs = np.empty((self.N_pixels, ), dtype=np.float32)
        self.nu_b_outs = np.empty((self.N_pixels, ), dtype=np.float32)
        self.mu_i_outs = np.empty((self.N_pixels, ), dtype=np.float32)
        self.nu_i_outs = np.empty((self.N_pixels, ), dtype=np.float32)
        self.mu_dot_outs = np.empty((self.N_pixels, ), dtype=np.float32)
        self.nu_dot_outs = np.empty((self.N_pixels, ), dtype=np.float32)
        self.raw_data = np.zeros((self.camera_height, self.camera_width, 8))
        self.is_animated = np.empty((self.N_pixels, ), dtype=np.int8)
        self.depth_upper_bound = np.empty((self.N_pixels,), dtype=np.float32)
        self.depth_lower_bound = np.empty((self.N_pixels,), dtype=np.float32)

        # cuda.memcpy_htod(self.d_animateds, self.is_animated)

    #@profile
    def run(self, args):
        """
        runs algo 1 to return a numpy array of shape
        (H, W), where each element is a boolean, representing
        whether the pixel is certified

        args = mu_i, nu_i, v_e, psi_e, a_e, phi_e
        """
        # unpack 
        mu_i, nu_i, v_e, psi_e, a_e, phi_e, is_animated, depth_upper_bound, depth_lower_bound = args

        # copy inputs into cuda device
        cuda.memcpy_htod(self.d_mus, mu_i)
        cuda.memcpy_htod(self.d_nus, nu_i)

        # setup arguments for the kernel
        # convert all numbers to np.number
        v_e = np.float32(v_e)
        psi_e = np.float32(psi_e)
        a_e = np.float32(a_e)
        phi_e = np.float32(phi_e)
        self.gridsearchsize = np.intc(self.gridsearchsize)
        self.N_pixels = np.intc(self.N_pixels)
        self.is_animated = np.int8(is_animated)
        self.depth_upper_bound = np.float32(depth_upper_bound)
        self.depth_lower_bound = np.float32(depth_lower_bound)
        # print("focal length: ", self.f)
        # print("Type of f: ")
        # print(type(self.f))

        cuda.memcpy_htod(self.d_animateds, self.is_animated)
        cuda.memcpy_htod(self.d_depth_upper_bound, self.depth_upper_bound)
        cuda.memcpy_htod(self.d_depth_lower_bound, self.depth_lower_bound)

        # call function

        pycuda.driver.Context.synchronize()
        self.certify_u_for_mu(self.f, self.d_mus, self.d_nus, self.d_Xs, self.d_Ys, self.d_offsets, v_e, psi_e, a_e,
                             phi_e, self.gridsearchsize, self.N_pixels, self.d_mu_is_certifieds, self.d_mu_b_outs,
                             self.d_mu_i_outs, self.d_mu_dot_outs, self.d_animateds, self.d_depth_upper_bound, 
                             self.d_depth_lower_bound, block=self.block_dim, grid=self.grid_dim)
        pycuda.driver.Context.synchronize()
        self.certify_u_for_nu(self.f, self.d_mus, self.d_nus, self.d_Xs, self.d_Ys, self.d_offsets, v_e, psi_e, a_e,
                             phi_e, self.gridsearchsize, self.N_pixels, self.d_nu_is_certifieds, self.d_nu_b_outs,
                             self.d_nu_i_outs, self.d_nu_dot_outs, self.d_animateds, self.d_depth_upper_bound, 
                             self.d_depth_lower_bound, block=self.block_dim, grid=self.grid_dim)
        pycuda.driver.Context.synchronize()

        # move results into host memory
        cuda.memcpy_dtoh(self.mu_is_certifieds, self.d_mu_is_certifieds)
        cuda.memcpy_dtoh(self.nu_is_certifieds, self.d_nu_is_certifieds)
        cuda.memcpy_dtoh(self.mu_b_outs, self.d_mu_b_outs)
        cuda.memcpy_dtoh(self.nu_b_outs, self.d_nu_b_outs)
        cuda.memcpy_dtoh(self.mu_i_outs, self.d_mu_i_outs)
        cuda.memcpy_dtoh(self.nu_i_outs, self.d_nu_i_outs)
        cuda.memcpy_dtoh(self.mu_dot_outs, self.d_mu_dot_outs)
        cuda.memcpy_dtoh(self.nu_dot_outs, self.d_nu_dot_outs)
        t1 = time.time()

        self.mu_is_certifieds = self.mu_is_certifieds.reshape((self.camera_height, self.camera_width))
        self.nu_is_certifieds = self.nu_is_certifieds.reshape((self.camera_height, self.camera_width))

        # self.d_upper = self

        # self.mu_b_outs = self.mu_b_outs.reshape((self.camera_height, self.camera_width))
        # self.nu_b_outs = self.nu_b_outs.reshape((self.camera_height, self.camera_width))
        # self.mu_i_outs = self.mu_i_outs.reshape((self.camera_height, self.camera_width))
        # self.nu_i_outs = self.nu_i_outs.reshape((self.camera_height, self.camera_width))
        # self.mu_dot_outs = self.mu_dot_outs.reshape((self.camera_height, self.camera_width))
        # self.nu_dot_outs = self.nu_dot_outs.reshape((self.camera_height, self.camera_width))

        self.raw_data[:, :, 0] = self.mu_is_certifieds.astype(np.float32)
        self.raw_data[:, :, 1] = self.nu_is_certifieds.astype(np.float32)
        self.raw_data[:, :, 2] = self.mu_b_outs.reshape((self.camera_height, self.camera_width))
        self.raw_data[:, :, 3] = self.nu_b_outs.reshape((self.camera_height, self.camera_width))
        self.raw_data[:, :, 4] = self.mu_i_outs.reshape((self.camera_height, self.camera_width))
        self.raw_data[:, :, 5] = self.nu_i_outs.reshape((self.camera_height, self.camera_width))
        self.raw_data[:, :, 6] = self.mu_dot_outs.reshape((self.camera_height, self.camera_width))
        self.raw_data[:, :, 7] = self.nu_dot_outs.reshape((self.camera_height, self.camera_width))

        # # visualize certified pixels
        # cv.imshow("certified mu", np.repeat(self.mu_is_certifieds[:, :, np.newaxis].astype(np.uint8) * 255, 3, axis=2))
        # cv.imshow("certified nu", np.repeat(self.nu_is_certifieds[:, :, np.newaxis].astype(np.uint8) * 255, 3, axis=2))
        if cv.waitKey(1) == 27:
            pass

        return np.logical_and(self.mu_is_certifieds, self.nu_is_certifieds), self.raw_data
        # return 0,0
