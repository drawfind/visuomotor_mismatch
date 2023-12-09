import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm

def main():
    experiment()

    
# Create a population code with N neurons, encoding the variable x
def code(x, N):
    xmin = -1.0
    xrange = 2.0
    sqrsigma = (xrange/5)**2
    code = np.zeros(N)
    
    for i in range(N):
        mean = xmin + i * xrange / (N - 1)  
        val = math.exp(- (x - mean)**2 / (2 * sqrsigma))
        code[i] = val
            
    return code


# Compute the activation of Layer 2/3 neurons
# given the speed of observed visual flow (observed_flow)
# and the speed predicted from locomotion (motor_based_predicted_flow)
def layer_activation(motor_based_predicted_flow, observed_flow, N):
    external_motion = observed_flow - motor_based_predicted_flow
    env_code = code(external_motion,N)
    return env_code


# Run the experiment
def experiment():
    N = 100 # Number of neurons in a population code
    num_trials = 10 # Number of different speed values
    threshold = 0.05 # Threshold for defining dMM and hMM neurons
    
    neuron_ind = [i for i in range(1, N + 1)]
    motor_flow = np.zeros(num_trials)
    sum_mismatch = np.zeros(N)
    all_mismatch = []
    
    for m in range(num_trials):
        motor_flow[m] = m*0.05 # Speed predicted from locomotion

        # Code1: Visual flow matches motor-induced flow
        env_code1 = layer_activation(motor_flow[m],motor_flow[m],N)
        # Code2: Visual flow is zero
        env_code2 = layer_activation(motor_flow[m],0,N)

        mismatch = env_code2 - env_code1
        sum_mismatch += mismatch
        all_mismatch.append(mismatch)
        

    # Plot neural activation by preferred speed comparing match (env_code1) and mismatch (env_code2) conditions
    preferred_speed = np.arange(0, N)/((N-1)/2)-1

    plt.figure(figsize=(8, 6))
    plt.scatter(preferred_speed, env_code1, label='Match', marker='o', facecolors='none', edgecolors='green')
    plt.plot(preferred_speed, env_code1, linestyle='-', color='green')
    plt.scatter(preferred_speed, env_code2, label='Mismatch', color='red')
    plt.plot(preferred_speed, env_code2, linestyle='-', color='red')
    plt.xlabel('Neuron (Preferred Speed)', fontsize=16)
    plt.ylabel('Activation', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True)

    fname = 'population_code.eps'
    #plt.savefig(fname, format='eps')
    plt.show()
    
    # Compute dMM and hMM neurons
    dMM_mot = []
    dMM_mis = []
    hMM_mot = []
    hMM_mis = []
       
    for m in range(num_trials):
        for i in range(N):
            if sum_mismatch[i] > threshold*num_trials:
                dMM_mot.append(motor_flow[m])
                dMM_mis.append(all_mismatch[m][i])
                
            if sum_mismatch[i] < -threshold*num_trials:
                hMM_mot.append(motor_flow[m])
                hMM_mis.append(all_mismatch[m][i])

    print('Number of dMM neurons:',len(dMM_mis)/num_trials)
    print('Number of hMM neurons:',len(hMM_mis)/num_trials)
    

    # Fit linear model to dMM data
    X = sm.add_constant(dMM_mot)
    model = sm.RLM(dMM_mis, X, M=sm.robust.norms.TukeyBiweight())
    result = model.fit()

    # Plot dMM data
    plt.figure(figsize=(8, 6))
    plt.scatter(dMM_mot, dMM_mis, label='dMM', marker='o', facecolors='none', edgecolors='brown')
    plt.plot(X[:,1], result.fittedvalues, 'k--')
    plt.xlabel('Locomotion Speed', fontsize=16)
    plt.ylabel('Mismatch Response', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('dMM Neurons', fontsize=16)
    plt.grid(True)

    fname = 'dMM_speed.eps'
    #plt.savefig(fname, format='eps')
    plt.show()

    # Fit linear model to hMM data
    X = sm.add_constant(hMM_mot)
    model = sm.RLM(hMM_mis, X, M=sm.robust.norms.TukeyBiweight())
    result = model.fit()

    # Plot hMM data
    plt.figure(figsize=(8, 6))
    plt.scatter(hMM_mot, hMM_mis, label='hMM', marker='o', facecolors='none', edgecolors='darkgreen')
    plt.plot(X[:,1], result.fittedvalues, 'k--')
    plt.xlabel('Locomotion Speed', fontsize=16)
    plt.ylabel('Mismatch Response', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('hMM Neurons', fontsize=16)
    plt.grid(True)
    
    fname = 'hMM_speed.eps'
    #plt.savefig(fname, format='eps')
    plt.show()



if __name__ == "__main__":
    main()
