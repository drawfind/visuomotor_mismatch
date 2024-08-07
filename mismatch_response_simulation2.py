# (c) 2023-2024 Heiko Hoffmann

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
    threshold = 0.02 # Threshold for defining dMM and hMM neurons
    num_pc = 3 # Number of different population codes
    offset_vec = np.array([1.07, 0.6]) # Offset of the velocity range of the population code
    
    neuron_ind = [i for i in range(1, N + 1)]
    motor_flow = np.zeros(num_trials*num_pc)
    sum_mismatch = np.zeros(N*num_pc)
    all_mismatch = []

    # Create basis vectors for the 1-d projections of the different population codes
    basis_vec = []
    for pc in range(num_pc):
        phi = 2*math.pi/num_pc * pc # Orientation of basis vector
        basis_vec.append([math.cos(phi), math.sin(phi)])
    
    for m in range(num_trials):
        speed = m*0.5/num_trials # Speed predicted from locomotion
        v_vec = np.array([speed, 0])
        motor_flow[m] = speed
        
        for pc in range(num_pc):
            
            # Compute projection of v_vec onto basis vector
            v = v_vec.dot(basis_vec[pc])

            # Compute projection of offset_vec onto basis vector
            offset_proj = offset_vec.dot(basis_vec[pc])
            
            # Code1: Visual flow matches motor-induced flow
            env_code1 = layer_activation(v,v,offset_proj,N)
            # Code2: Visual flow is zero
            env_code2 = layer_activation(v,0,offset_proj,N)

            mismatch = env_code2 - env_code1

            if pc==0:
                mismatch_neurons = mismatch
            else:
                mismatch_neurons = np.concatenate((mismatch_neurons, mismatch))

        sum_mismatch += mismatch_neurons
        all_mismatch.append(mismatch_neurons)
        
    
    # Compute dMM and hMM neurons
    dMM_mot = []
    dMM_mis = []
    hMM_mot = []
    hMM_mis = []
    dMM_ind = []
    hMM_ind = []
       
    for m in range(num_trials):
        for i in range(N*num_pc):
            if sum_mismatch[i] > threshold*num_trials:
                dMM_mot.append(motor_flow[m])
                dMM_mis.append(all_mismatch[m][i])
                dMM_ind.append(i) # keep track of neural index
                
            if sum_mismatch[i] < -threshold*num_trials:
                hMM_mot.append(motor_flow[m])
                hMM_mis.append(all_mismatch[m][i])
                hMM_ind.append(i) # keep track of neural index

    nd = len(dMM_mis)/num_trials
    nh = len(hMM_mis)/num_trials
    
    print('Number of dMM neurons:', nd)
    print('Number of hMM neurons:', nh)
    print('Number of unclassified neurons:', N*num_pc - nd - nh)
    
    # Fit linear model to dMM data
    X = sm.add_constant(dMM_mot)
    model = sm.RLM(dMM_mis, X, M=sm.robust.norms.TukeyBiweight())
    result = model.fit()

    # Plot dMM data
    plt.figure(figsize=(8, 6))
    plt.scatter(dMM_mot, dMM_mis, label='dMM', marker='o', facecolors='none', edgecolors='brown')

    # Connect points with the same index
    unique_indices = set(dMM_ind)
    for idx in unique_indices:
        x_points = [dMM_mot[i] for i in range(len(dMM_mot)) if dMM_ind[i] == idx]
        y_points = [dMM_mis[i] for i in range(len(dMM_mis)) if dMM_ind[i] == idx]
        plt.plot(x_points, y_points, color='brown')
    
    #plt.plot(X[:,1], result.fittedvalues, 'k--')
    plt.xlabel('Locomotion Speed', fontsize=16)
    plt.ylabel('Mismatch Response', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('dMM Neurons', fontsize=16)
    plt.grid(True)

    fname = 'dMM_speed_3pcs.eps'
    plt.savefig(fname, format='eps')
    plt.show()

    # Fit linear model to hMM data
    X = sm.add_constant(hMM_mot)
    model = sm.RLM(hMM_mis, X, M=sm.robust.norms.TukeyBiweight())
    result = model.fit()

    # Plot hMM data
    plt.figure(figsize=(8, 6))
    plt.scatter(hMM_mot, hMM_mis, label='hMM', marker='o', facecolors='none', edgecolors='darkgreen')

    # Connect points with the same index
    unique_indices = set(hMM_ind)
    for idx in unique_indices:
        x_points = [hMM_mot[i] for i in range(len(hMM_mot)) if hMM_ind[i] == idx]
        y_points = [hMM_mis[i] for i in range(len(hMM_mis)) if hMM_ind[i] == idx]
        plt.plot(x_points, y_points, color='darkgreen')
    
    #plt.plot(X[:,1], result.fittedvalues, 'k--')
    plt.xlabel('Locomotion Speed', fontsize=16)
    plt.ylabel('Mismatch Response', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('hMM Neurons', fontsize=16)
    plt.grid(True)
    
    fname = 'hMM_speed_3pcs.eps'
    plt.savefig(fname, format='eps')
    plt.show()



if __name__ == "__main__":
    main()
