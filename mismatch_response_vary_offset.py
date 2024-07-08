# (c) 2024 Heiko Hoffmann

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm

def main():
    experiment()

    
# Create a population code with N neurons, encoding the variable x.
# X_offset shifts the encoded range.
def code(x, x_offset, N):
    xmin = -1.0
    xrange = 2.0
    sqrsigma = (xrange/5)**2
    code = np.zeros(N)
    
    for i in range(N):
        mean = xmin + i * xrange / (N - 1) - x_offset
        val = math.exp(- (x - mean)**2 / (2 * sqrsigma))
        code[i] = val
            
    return code


# Compute the activation of Layer 2/3 neurons
# given the speed of observed visual flow (observed_flow)
# and the speed predicted from locomotion (motor_based_predicted_flow)
def layer_activation(motor_based_predicted_flow, observed_flow, v_offset, N):
    external_motion = observed_flow - motor_based_predicted_flow
    env_code = code(external_motion, v_offset, N)
    return env_code


# Run the experiment
def experiment():
    N = 100 # Number of neurons in a population code
    num_trials = 10 # Number of different speed values
    threshold = 0.05 # Threshold for defining dMM and hMM neurons

    n_steps = 101

    vp = np.zeros(n_steps)
    nd = np.zeros(n_steps)
    nh = np.zeros(n_steps)
    nu = np.zeros(n_steps)
    
    for k in range(n_steps):
        v_offset = -1.0 + 2.0 * k / (n_steps - 1)  
        
        sum_mismatch = np.zeros(N)

        for m in range(num_trials):
            motor_flow = m*0.5/num_trials # Speed predicted from locomotion

            # Code1: Visual flow matches motor-induced flow
            env_code1 = layer_activation(motor_flow,motor_flow,v_offset,N)
            # Code2: Visual flow is zero
            env_code2 = layer_activation(motor_flow,0,v_offset,N)

            mismatch = env_code2 - env_code1
            sum_mismatch += mismatch
       
        # Compute dMM and hMM neurons
        dMM_mis = 0
        hMM_mis = 0
       
        for m in range(num_trials):
            for i in range(N):
                if sum_mismatch[i] > threshold*num_trials:
                    dMM_mis += 1
                
                if sum_mismatch[i] < -threshold*num_trials:
                    hMM_mis += 1

        vp[k] = v_offset
        nd[k] = dMM_mis/num_trials
        nh[k] = hMM_mis/num_trials
        nu[k] = N - nd[k] - nh[k]

    plt.figure(figsize=(8, 6))
    # Plot each vector separately with fill_between and specify hatch pattern
    plt.fill_between(vp, nd, color='brown', alpha=0.5, label='dMM', hatch='//')
    plt.fill_between(vp, nd, nd+nh, color='darkgreen', alpha=0.5, label='hMM', hatch='+')
    plt.fill_between(vp, nd+nh, N, color='gray', alpha=0.5, label='Unclassified', hatch='\\')
    
    plt.legend(loc='lower right', fontsize=16)
    plt.xlabel('Offset of Speed Range [Same unit as in A]', fontsize=16)
    plt.ylabel('Number of Neurons', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Rasterize the parts of the plot containing the hatch patterns
    plt.gcf().set_rasterized(True)
    fname = 'number_neurons_voffset.eps'
    plt.savefig(fname, format='eps')

    plt.show()

if __name__ == "__main__":
    main()
