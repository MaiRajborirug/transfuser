
#define PI 3.141592654f
#define V_I_MIN 0.0f 
#define V_I_MAX 10.0f  // -V

#define A_MIN -10.0f // -A
#define A_MAX 6.0f

#define PSI_I_MIN -0.1f
#define PSI_I_MAX 0.1f

// radius to keep obstacles out of, in meters
#define R 1.0f
#define ALPHA1 1.0f
#define ALPHA2 1.0f

// height threshold for obstacle avoidance
// this is down positive, origin at camera location
#define H_BAR 2.3f // ~ position at the floor

#define IMG_W 960
#define IMG_H 720


__device__ 
float D_mu(
    float f, 
    float mu_i, 
    float X_i_t,
    float v_i,
    float v_e,
    float theta, 
    float psi_e, 
    float CHI)
{
    float nom = 
    f * (X_i_t * v_e - X_i_t * v_i * cosf(theta) + f * v_i * sinf(theta));
    float denom = 
    CHI * (psi_e * X_i_t * X_i_t + psi_e * f * f - mu_i * f);
    return nom / denom;
}

__device__         
void solve_eqn_43_for_theta(
    float f, 
    float X_i_t, 
    float* D_theta_candidates
){
    float theta = atan2f(-f, X_i_t);
    D_theta_candidates[0] = theta;
    D_theta_candidates[1] = fmodf(theta + PI, 2 * PI);
}

__device__
void optimize_D_mu(
    float f, 
    float mu_i, 
    float X_i_t, 
    float v_e, 
    float psi_e, 
    float CHI,
    float* D_max,
    float* D_min,
    int8_t animated
){
    float D_theta_candidates[2];
    solve_eqn_43_for_theta(f, X_i_t, D_theta_candidates);
    // V_I_MIN must be zero
    float candidates[3];

    // D_mu is constant w.r.t theta when v = 0

    candidates[0] = D_mu(
        f, 
        mu_i, 
        X_i_t, 
        V_I_MIN * animated,
        v_e,
        D_theta_candidates[0],
        psi_e, 
        CHI);

    candidates[1] = D_mu(
        f, 
        mu_i, 
        X_i_t, 
        V_I_MAX * animated,
        v_e,
        D_theta_candidates[0],
        psi_e, 
        CHI);

    candidates[2] = D_mu(
        f, 
        mu_i, 
        X_i_t, 
        V_I_MAX * animated,
        v_e,
        D_theta_candidates[1],
        psi_e, 
        CHI);

    
    bool first_iter = true;
    float max;
    float min;
    for (int i = 0; i < 3; i++){
        if (first_iter){
            max = candidates[i];
            min = candidates[i];
            first_iter = false;
        } else {
            if (candidates[i] > max){
                max = candidates[i];
            } else if (candidates[i] < min) {
                min = candidates[i];
            }
        }
    }

    //*D_max = max;
    if (max > R){
        *D_max = max;
    } else {
        *D_max = R;
    }
    if (min > R){
        *D_min = min;
    } else {
        *D_min = R;
    }
}

__device__
float D_nu(
    float f, 
    float nu_i, 
    float X_i_t,
    float Y_i_t, 
    float v_i,
    float v_e,
    float theta, 
    float psi_e, 
    float CHI)
{   
    float nom =
    Y_i_t * f * (v_e - v_i * cosf(theta));
    float denom = 
    CHI * (f * nu_i - X_i_t * Y_i_t * psi_e);
    return nom / denom;
}

__device__ 
void optimize_D_nu(
    float f, 
    float nu_i, 
    float X_i_t, 
    float Y_i_t,
    float v_e, 
    float psi_e, 
    float CHI,
    float* D_max,
    float* D_min,
    int8_t animated
){
    float theta_candidate_1 = 0.0f;
    float theta_candidate_2 = PI;
    float candidates[3];

    candidates[0] = D_nu(
        f, 
        nu_i, 
        X_i_t,
        Y_i_t, 
        0.0f,
        v_e,
        theta_candidate_1, 
        psi_e, 
        CHI);

    candidates[1] = D_nu(
        f, 
        nu_i, 
        X_i_t,
        Y_i_t, 
        V_I_MAX * animated,
        v_e,
        theta_candidate_1, 
        psi_e, 
        CHI);

    candidates[2] = D_nu(
        f, 
        nu_i, 
        X_i_t,
        Y_i_t, 
        V_I_MAX * animated,
        v_e,
        theta_candidate_2, 
        psi_e, 
        CHI);

    bool first_iter = true;
    float max;
    float min;
    for (int i = 0; i < 3; i++){
        if (first_iter){
            max = candidates[i];
            min = candidates[i];
            first_iter = false;
        } else {
            if (candidates[i] > max){
                max = candidates[i];
            } else if (candidates[i] < min) {
                min = candidates[i];
            }
        }
    }

    //*D_max = max;
    if (max > R){
        *D_max = max;
    } else {
        *D_max = R;
    }
    if (min > R){
        *D_min = min;
    } else {
        *D_min = R;
    }
}

__device__
void calculate_D_bounds(
    float f, 
    float mu_i,        
    float nu_i, 
    float X_i_t,
    float Y_i_t, 
    float v_e, 
    float psi_e, 
    float CHI,
    float* D_min,
    float* D_max,
    int8_t animated
    )
{
    float D_min_candidate_mu = 0.0f;
    float D_max_candidate_mu = 0.0f;
    float D_min_candidate_nu = 0.0f;
    float D_max_candidate_nu = 0.0f;

    optimize_D_mu(
        f, 
        mu_i, 
        X_i_t, 
        v_e, 
        psi_e, 
        CHI,
        &D_max_candidate_mu,
        &D_min_candidate_mu,
        animated
    );

    optimize_D_nu(
        f, 
        nu_i, 
        X_i_t, 
        Y_i_t,
        v_e, 
        psi_e, 
        CHI,
        &D_max_candidate_nu,
        &D_min_candidate_nu,
        animated
    );

    *D_min = fmaxf(D_min_candidate_mu, D_min_candidate_nu);
    *D_max = fminf(D_max_candidate_mu, D_max_candidate_nu);
}


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
    float term5 = a_i * f * sinf(theta); 
    float term6 = -f * psi_e * v_e;
    float COS_THETA = cosf(theta);
    float term7 = -X_i_t * a_i * COS_THETA;
    float term8 = 2 * f * psi_e * v_i * COS_THETA;
    float term9 = -f * psi_i * v_i * COS_THETA;
    float term10 = sinf(theta) * (2 * X_i_t * psi_e * v_i - X_i_t * psi_i * v_i);

    return term1 + term2 + term3 + 
        (term4 + term5 + term6 + term7 + term8 + term9 + term10) / D_CHI;
    //return f;
}

// __device__
// void solve_eqn_60_for_0(float f, float X_i_t, float* theta_candidates){
//     float theta = atanf(X_i_t / f);
//     theta_candidates[0] = theta;
//     theta_candidates[1] = fmodf(theta + PI, 2 * PI);
// }

// __device__
// void solve_theta_guess_from_59(float f, float X_i_t, float* theta_candidates){
//     float theta = atanf(-f / X_i_t);
//     theta_candidates[2] = theta;
//     theta_candidates[3] = fmodf(theta + PI, 2 * PI);
// }

// __device__
// void solve_eqn_59_for_0(float f, float X_i_t, float* theta_candidates){
//     solve_theta_guess_from_59(f, X_i_t, theta_candidates);
// }

__device__
float optimize_mu_dot_i(
            float f, 
            float mu_i,
            float nu_i, 
            float X_i_t, 
            float Y_i_t, 
            float v_e, 
            float psi_e, 
            float a_e, 
            float phi_e,
            unsigned int N,
            bool findmax,
            int8_t animated,
            float d_upper_bound,
            float d_lower_bound)
{
    // iterate over boundary values of a_i, v_i, psi_i and D,
    // brute force over theta

    // compute CHI
    // inverse squre root
    float CHI = rsqrtf(((X_i_t * X_i_t) / (f * f)) + 1.0f);

    // float D_MAX = 0.0f;
    // float D_MIN = 0.0f;
    // calculate_D_bounds(
    //     f, 
    //     mu_i,        
    //     nu_i, 
    //     X_i_t,
    //     Y_i_t, 
    //     v_e, 
    //     psi_e, 
    //     CHI,
    //     &D_MIN,
    //     &D_MAX,
    //     animated); // assign D_MIN, D_MAX new values
    float a_i_candidates[2] = {A_MIN, A_MAX};
    float v_i_candidates[2] = {V_I_MIN, V_I_MAX};
    float psi_i_candidates[2] = {PSI_I_MIN, PSI_I_MAX};
    float D_candidates[2] ={d_upper_bound, d_lower_bound};
    float theta_candidate = 0.0f;
    float theta_stepsize = 2.0f * PI / (float)N;

    float best_y;
    bool first_iter = true; 
    float candidate_y;

    // non animate and animate objects
    if (animated){
        for (int a_i_idx = 0; a_i_idx < 2; a_i_idx ++){
            for (int v_i_idx = 0; v_i_idx < 2; v_i_idx ++){
                for (int psi_i_idx = 0; psi_i_idx < 2; psi_i_idx ++){
                    for (int D_idx = 0; D_idx < 2; D_idx ++){
                        float a_i_candidate = a_i_candidates[a_i_idx];
                        float v_i_candidate = v_i_candidates[v_i_idx];
                        float psi_i_candidate = psi_i_candidates[psi_i_idx];
                        float D_candidate = D_candidates[D_idx];

                        for (int i = 0; i < N; i++){
                            candidate_y = eqn_54(a_i_candidate,
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

                            if (first_iter){
                                best_y = candidate_y;
                                first_iter = false;
                            }
                            else if (findmax && best_y < candidate_y){
                                best_y = candidate_y;
                            }else if (!findmax && best_y > candidate_y){
                                best_y = candidate_y;
                            }
                            theta_candidate += theta_stepsize;
                        }
                    }
                }
            }
        }
    }
    else {
        // for (int D_idx = 0; D_idx < 2; D_idx ++){
            // float D_candidate = D_candidates[D_idx];
        float D_candidate = d_upper_bound;

        for (int i = 0; i < N; i++){
            candidate_y = eqn_54(0.0f, //a_i_candidate,
                            f, 
                            mu_i, 
                            nu_i, 
                            X_i_t,
                            Y_i_t,
                            0.0f, //v_i_candidate, 
                            v_e, 
                            0.0f, //psi_i_candidate,
                            psi_e,
                            a_e,
                            theta_candidate,
                            phi_e,
                            CHI, 
                            D_candidate);

            if (first_iter){
                best_y = candidate_y;
                first_iter = false;
            }
            else if (findmax && best_y < candidate_y){
                best_y = candidate_y;
            }else if (!findmax && best_y > candidate_y){
                best_y = candidate_y;
            }
            theta_candidate += theta_stepsize;
        }
        // }
    }
    return best_y;
    //return D_MAX;
}

__global__
void certify_u_for_mu(
            float f, 
            float* mu_is,
            float* nu_is, 
            float* X_i_ts, 
            float* Y_i_ts,
            float* offsets,
            float v_e, 
            float psi_e, 
            float a_e, 
            float phi_e,
            unsigned int N_points,
            unsigned int N_pixels,
            bool* u_certified_for_mu,
            float* mu_b_out,
            float* mu_i_out,
            float* mu_dot_out,
            int8_t* animateds,
            float* d_upper_bounds,
            float* d_lower_bounds){

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = blockId * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    if (tid >= IMG_H * IMG_W){
        return;
    }

    float mu_i = mu_is[tid];
    float nu_i = nu_is[tid]; 
    float X_i_t = X_i_ts[tid];
    float Y_i_t = Y_i_ts[tid];
    float offset = offsets[tid];
    int8_t animated = animateds[tid];
    float d_upper_bound = d_upper_bounds[tid];
    float d_lower_bound = d_lower_bounds[tid];

    float mu_b = R * R * Y_i_t * nu_i / (X_i_t * H_BAR * H_BAR);
    mu_b_out[tid] = mu_b;
    mu_i_out[tid] = mu_i;
    bool mu_find_upperbound = X_i_t < 0.0f;

    if (mu_find_upperbound && mu_i >= mu_b){
        float mu_upper = optimize_mu_dot_i(
            f, 
            mu_i,
            nu_i, 
            X_i_t, 
            Y_i_t, 
            v_e, 
            psi_e, 
            a_e, 
            phi_e, 
            N_points,
            true,
            animated,
            d_upper_bound,
            d_lower_bound);
        u_certified_for_mu[tid] = mu_upper < -5000.0f*offset;
        mu_dot_out[tid] = mu_upper;
    } else if (!mu_find_upperbound && mu_i <= mu_b){
        float mu_lower = optimize_mu_dot_i(
            f, 
            mu_i,
            nu_i, 
            X_i_t, 
            Y_i_t, 
            v_e, 
            psi_e, 
            a_e, 
            phi_e, 
            N_points,
            false,
            animated,
            d_upper_bound,
            d_lower_bound);
        u_certified_for_mu[tid] = mu_lower > 5000.0f*offset;
        mu_dot_out[tid] = mu_lower;
    } else {
        u_certified_for_mu[tid] = true;
        mu_dot_out[tid] = 0.0f;
    }

    
}

__device__
float eqn_55(float a_i,
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
    float term1 = Y_i_t * psi_e * psi_e;
    float D_CHI = D * CHI;
    float term2 = 2.0f * Y_i_t * v_i * sinf(theta) * psi_e / D_CHI;
    float term3 = 2.0f * nu_i * nu_i / Y_i_t;
    float term4 = Y_i_t * a_e / D_CHI;
    float term5 = X_i_t * Y_i_t * phi_e / f;
    float term6 = -Y_i_t * a_i * cosf(theta) / D_CHI;
    float term7 = -Y_i_t * psi_i * v_i * sinf(theta) / D_CHI;
    return term1 + term2 + term3 + term4 + term5 + term6 + term7;
    //return term1;
}

__device__ 
float optimize_nu_dot_i(
            float f, 
            float mu_i,
            float nu_i, 
            float X_i_t, 
            float Y_i_t, 
            float v_e, 
            float psi_e, 
            float a_e, 
            float phi_e, 
            unsigned int N, 
            bool findmax,
            int8_t animated,
            float d_upper_bound,
            float d_lower_bound){

    // brute force over theta,
    // use boundary values for a_i, v_i, psi_e and D
    // since mu_dot is monotonomus w.r.t these 4 variables

    
     // iterate over boundary values of a_i, v_i, psi_i and D,
    // brute force over theta

    // compute CHI
    // inverse squre root

    float CHI = rsqrtf(((X_i_t * X_i_t) / (f * f)) + 1.0f);

    // float D_MIN = 0.0f;
    // float D_MAX = d_depth_array;
    // calculate_D_bounds(
    //     f, 
    //     mu_i,        
    //     nu_i, 
    //     X_i_t,
    //     Y_i_t, 
    //     v_e, 
    //     psi_e, 
    //     CHI,
    //     &D_MIN,
    //     &D_MAX,
    //     animated);

    float a_i_candidates[2] = {A_MIN, A_MAX};
    float v_i_candidates[2] = {V_I_MIN, V_I_MAX};
    float psi_i_candidates[2] = {PSI_I_MIN, PSI_I_MAX};
    float D_candidates[2] ={d_upper_bound, d_lower_bound};
    float theta_candidate = 0.0f;
    float theta_stepsize = 2.0f * PI / (float)N;

    float best_y;
    bool first_iter = true; 
    float candidate_y;

    // non animate and animate objects
    if (animated){
        for (int a_i_idx = 0; a_i_idx < 2; a_i_idx ++){
            for (int v_i_idx = 0; v_i_idx < 2; v_i_idx ++){
                for (int psi_i_idx = 0; psi_i_idx < 2; psi_i_idx ++){
                    for (int D_idx = 0; D_idx < 2; D_idx ++){
                        float a_i_candidate = a_i_candidates[a_i_idx];
                        float v_i_candidate = v_i_candidates[v_i_idx];
                        float psi_i_candidate = psi_i_candidates[psi_i_idx];
                        float D_candidate = D_candidates[D_idx];

                        for (int i = 0; i < N; i++){
                            candidate_y = eqn_54(a_i_candidate,
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

                            if (first_iter){
                                best_y = candidate_y;
                                first_iter = false;
                            }
                            else if (findmax && best_y < candidate_y){
                                best_y = candidate_y;
                            }else if (!findmax && best_y > candidate_y){
                                best_y = candidate_y;
                            }
                            theta_candidate += theta_stepsize;
                        }
                    }
                }
            }
        }
    }
    else {
        // for (int D_idx = 0; D_idx < 2; D_idx ++){
            // float D_candidate = D_candidates[D_idx];
        float D_candidate = d_upper_bound;

        for (int i = 0; i < N; i++){
            candidate_y = eqn_54(0.0f, //a_i_candidate,
                            f, 
                            mu_i, 
                            nu_i, 
                            X_i_t,
                            Y_i_t,
                            0.0f, //v_i_candidate, 
                            v_e, 
                            0.0f, //psi_i_candidate,
                            psi_e,
                            a_e,
                            theta_candidate,
                            phi_e,
                            CHI, 
                            D_candidate);

            if (first_iter){
                best_y = candidate_y;
                first_iter = false;
            }
            else if (findmax && best_y < candidate_y){
                best_y = candidate_y;
            }else if (!findmax && best_y > candidate_y){
                best_y = candidate_y;
            }
            theta_candidate += theta_stepsize;
        }
        // }
    }
    return best_y;
}


__global__
void certify_u_for_nu(
            float f, 
            float* mu_is,
            float* nu_is, 
            float* X_i_ts, 
            float* Y_i_ts, 
            float* offsets,
            float v_e, 
            float psi_e, 
            float a_e, 
            float phi_e, 
            unsigned int N_points,
            unsigned int N_pixels,
            bool* u_certified_for_nu,
            float* nu_b_out,
            float* nu_i_out,
            float* nu_dot_out,
            int8_t* animateds,
            float* d_upper_bounds,
            float* d_lower_bounds){

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = blockId * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;

    
    if (tid >= IMG_H * IMG_W){
        return;
    } 
    
    float mu_i = mu_is[tid];
    float nu_i = nu_is[tid]; 
    float X_i_t = X_i_ts[tid];
    float Y_i_t = Y_i_ts[tid]; 
    float offset = offsets[tid];
    int8_t animated = animateds[tid];
    float d_upper_bound = d_upper_bounds[tid];
    float d_lower_bound = d_lower_bounds[tid];

    float nu_b = H_BAR * H_BAR * X_i_t * mu_i / (R * R * Y_i_t);
    nu_b_out[tid] = nu_b;
    nu_i_out[tid] = nu_i;
    bool nu_find_upperbound = true;

    if (nu_find_upperbound && nu_i >= nu_b){
        float nu_upper = optimize_nu_dot_i(
            f, 
            mu_i,
            nu_i, 
            X_i_t, 
            Y_i_t, 
            v_e, 
            psi_e, 
            a_e, 
            phi_e, 
            N_points,
            true,
            animated,
            d_upper_bound,
            d_lower_bound);
        u_certified_for_nu[tid] = nu_upper < -offset;
        nu_dot_out[tid] = nu_upper;
    } else if (!nu_find_upperbound && nu_i <= nu_b){
        float nu_lower = optimize_nu_dot_i(
            f, 
            mu_i,
            nu_i, 
            X_i_t, 
            Y_i_t, 
            v_e, 
            psi_e, 
            a_e, 
            phi_e, 
            N_points,
            false,
            animated,
            d_upper_bound,
            d_lower_bound);
        u_certified_for_nu[tid] = nu_lower > offset;
        nu_dot_out[tid] = nu_lower;
    } else {
        u_certified_for_nu[tid] = true;
        nu_dot_out[tid] = 0.0f;
    }

}
