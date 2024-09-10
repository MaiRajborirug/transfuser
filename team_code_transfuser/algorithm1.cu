// this must be the first include!
// https://docs.python.org/3.8/extending/extending.html

#include <Python.h>

#include <math.h>
#include <iostream>
#include <chrono>
#define PY_SSIZE_T_CLEAN


#define PI 3.141592654f
#define D_MIN -1.0f
#define D_MAX 1.0f
#define V_I_MIN -1.0f
#define V_I_MAX 1.0f
#define A_MIN -1.0f
#define A_MAX 1.0f
#define PSI_I_MIN -1.0f
#define PSI_I_MAX 1.0f

int N_pixels = 1000000;

float* fs;
float* mu_is;
float* nu_is; 
float* X_i_ts; 
float* Y_i_ts; 
float* v_es; 
float* psi_es; 
float* a_es; 
float* phi_es; 
float* CHIs;

unsigned int N_points = 500;
float* best_a_is;
float* best_Ds;
float* best_psi_is;
float* best_thetas;
float* best_v_is;

unsigned int blockSize, numBlocks;



__device__
float eqn_54(float a_i,
            float f, 
            float mu_i, 
            float nu_i, 
            float X_i_t, 
            float Y_i_t, 
            float v_i, 
            float v_e, 
            float psi_i,
            float psi_e, 
            float a_e,
            float theta, 
            float phi_e, 
            float CHI, 
            float D)
{
    float term1 = phi_e * f;
    float term2 = 2 * mu_i * nu_i / Y_i_t;
    float term3 = (X_i_t * X_i_t) * phi_e / f;
    float D_CHI = D * CHI;
    float term4 = X_i_t * a_e;
    float term5 = a_i * f * sin(theta); 
    float term6 = -f * psi_e * v_e;
    float COS_THETA = cos(theta);
    float term7 = -X_i_t * a_i * COS_THETA;
    float term8 = 2 * f * psi_e * v_i * COS_THETA;
    float term9 = -f * psi_i * v_i * COS_THETA;
    float term10 = sin(theta) * 2 * X_i_t * psi_e * v_i - X_i_t * psi_i * v_i;

    return term1 + term2 + term3 + 
        (term4 + term5 + term6 + term7 + term8 + term9 + term10) / D_CHI;
}

__device__
void solve_eqn_60_for_0(float f, float X_i_t, float* theta_candidates){
    float theta = atanf(X_i_t / f);
    theta_candidates[0] = theta;
    theta_candidates[1] = fmodf(theta + PI, 2 * PI);
}

__device__
void solve_theta_guess_from_59(float f, float X_i_t, float* theta_candidates){
    float theta = atanf(-f / X_i_t);
    theta_candidates[2] = theta;
    theta_candidates[3] = fmodf(theta + PI, 2 * PI);
}

__device__
void solve_eqn_59_for_0(float f, float X_i_t, float* theta_candidates){
    solve_theta_guess_from_59(f, X_i_t, theta_candidates);
}

__global__
void optimize(float* fs, 
            float* mu_is,
            float* nu_is, 
            float* X_i_ts, 
            float* Y_i_ts, 
            float* v_es, 
            float* psi_es, 
            float* a_es, 
            float* phi_es, 
            float* CHIs,
            unsigned int N,
            float* best_a_is,
            float* best_Ds,
            float* best_psi_is,
            float* best_thetas,
            float* best_v_is)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 100000)
        return;
        
    float f = fs[tid];  
    float mu_i = mu_is[tid];
    float nu_i = nu_is[tid]; 
    float X_i_t = X_i_ts[tid];
    float Y_i_t = Y_i_ts[tid]; 
    float v_e = v_es[tid];
    float psi_e = psi_es[tid];
    float a_e = a_es[tid];
    float phi_e = phi_es[tid];
    float CHI = CHIs[tid];

    float theta_candidates[4];
    float v_i_candidates[3] = {V_I_MIN, 0.0f, V_I_MAX};
    float psi_i_candidates[3] = {PSI_I_MIN, psi_e * 2, PSI_I_MAX};
    float D_candidates[2] ={D_MIN, D_MAX};

    solve_eqn_60_for_0(f, X_i_t, theta_candidates);
    solve_eqn_59_for_0(f, X_i_t, theta_candidates);

    float stepsize = (A_MAX - A_MIN) / N;
    float min_y = -INFINITY;
    float candidate_y;

    for (int theta_idx = 0; theta_idx < 4; theta_idx ++){
        for (int v_i_idx = 0; v_i_idx < 3; v_i_idx ++){
            for (int psi_i_idx = 0; psi_i_idx < 3; psi_i_idx ++){
                float candidate_a_i = A_MIN;
                for (int D_idx = 0; D_idx < 2; D_idx ++){

                    float theta_candidate = theta_candidates[theta_idx];
                    float v_i_candidate = v_i_candidates[v_i_idx];
                    float psi_i_candidate = psi_i_candidates[psi_i_idx];
                    float D_candidate = D_candidates[D_idx];

                    for (int i = 0; i < N; i++){
                        candidate_y = eqn_54(candidate_a_i,
                                        f, 
                                        mu_i, 
                                        nu_i, 
                                        X_i_t,
                                        Y_i_t,
                                        v_i_candidate, 
                                        v_e, 
                                        psi_i_candidate,
                                        psi_e,
                                        a_e,
                                        theta_candidate,
                                        phi_e,
                                        CHI, 
                                        D_candidate);

                        if (min_y == -INFINITY || min_y > candidate_y){
                            min_y = candidate_y;
                            best_a_is[tid] = candidate_a_i;
                            best_thetas[tid] = theta_candidate;
                            best_Ds[tid] = D_candidate;
                            best_v_is[tid] = v_i_candidate;
                            best_psi_is[tid] = psi_i_candidate;
                        }
                        candidate_a_i += stepsize;
                    }
                }
            }
        }
    }
}
