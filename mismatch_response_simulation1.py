import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import statsmodels.api as sm
import seaborn as sns

def main():
    experiment()


class PopulationCode():
    def __init__(self, x_offset, N, uniform):
        self.N = N
        self.sqrsigma = (2.0/5)**2
        if uniform:
            self.mean = np.linspace(-1 - x_offset, 1 - x_offset, N)
        else:
            self.mean = np.random.uniform(-1 - x_offset, 1 - x_offset, N)
        

    # Create a population code with N neurons, encoding the variable x.
    # X_offset shifts the encoded range.
    def encode(self,x):
        code = np.zeros(self.N)
    
        for i in range(self.N):
            val = math.exp(- (x - self.mean[i])**2 / (2 * self.sqrsigma))
            code[i] = val
            
        return code

    def get_preferred_speed(self):
        return self.mean
    

# Compute the activation of Layer 2/3 neurons
# given the speed of observed visual flow (observed_flow)
# and the speed predicted from locomotion (motor_based_predicted_flow)
def layer_activation(motor_based_predicted_flow, observed_flow, population_code):
    external_motion = observed_flow - motor_based_predicted_flow
    env_code = population_code.encode(external_motion)
    return env_code


# Run the experiment
def experiment():
    N = 100 # Number of neurons in a population code
    num_trials = 10 # Number of different speed values
    num_repeat = 20 # Number of repetitions, for computing correlation
                    # with added Gaussian noise
    threshold = 0.05 # Threshold for defining dMM and hMM neurons
    v_offset = 0.76 # Offset of the velocity range of the population code
    uniform_code = True # Preferred neural values have uniform spacing.
                        # Use 'False' for random spacing
    
    population_code = PopulationCode(v_offset, N, uniform_code)
    neuron_ind = [i for i in range(1, N + 1)]
    motor_flow = np.zeros(num_trials)
    sum_mismatch = np.zeros(N)
    all_mismatch = []
    
    for m in range(num_trials):
        motor_flow[m] = m*0.5/num_trials # Speed predicted from locomotion

        # Code1: Visual flow matches motor-induced flow
        env_code1 = layer_activation(motor_flow[m], motor_flow[m], population_code)
        # Code2: Visual flow is zero
        env_code2 = layer_activation(motor_flow[m], 0, population_code)

        mismatch = env_code2 - env_code1
        sum_mismatch += mismatch
        all_mismatch.append(mismatch)
        

    # Plot neural activation by preferred speed comparing match (env_code1) and mismatch (env_code2) conditions
    preferred_speed = population_code.get_preferred_speed()

    plt.figure(figsize=(8, 6))
    plt.scatter(preferred_speed, env_code1, label='Match', marker='o', facecolors='none', edgecolors='green')
    plt.scatter(preferred_speed, env_code2, label='Mismatch', color='red')
    if uniform_code:
        plt.plot(preferred_speed, env_code1, linestyle='-', color='green')
        plt.plot(preferred_speed, env_code2, linestyle='-', color='red')
    plt.xlabel('Preferred Speed Difference (Neuron) [Arbitrary unit]', fontsize=16)
    plt.ylabel('Activation [Scaled between 0 and 1]', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True)

    fname = 'population_code.eps'
    plt.savefig(fname, format='eps')
    plt.show()
    
    # Compute dMM and hMM neurons
    dMM_mot = []
    dMM_mis = []
    hMM_mot = []
    hMM_mis = []
    dMM_ind = []
    hMM_ind = []

    for m in range(num_trials):
        for i in range(N):
            if sum_mismatch[i] > threshold*num_trials:
                dMM_mot.append(motor_flow[m])
                dMM_mis.append(all_mismatch[m][i])
                dMM_ind.append(i) # keep track of neural index
                
            if sum_mismatch[i] < -threshold*num_trials:
                hMM_mot.append(motor_flow[m])
                hMM_mis.append(all_mismatch[m][i])
                hMM_ind.append(i) # keep track of neural index

    nd = int(len(dMM_mis)/num_trials)
    nh = int(len(hMM_mis)/num_trials)
    nu = int(N - nd - nh)
                
    print('Number of dMM neurons:', nd)
    print('Number of hMM neurons:', nh)
    print('Number of unclassified neurons:',nu)
 
    # Fit linear model to dMM data
    X = sm.add_constant(dMM_mot)
    model = sm.RLM(dMM_mis, X, M=sm.robust.norms.TukeyBiweight())
    result = model.fit()
    print("=====================")
    print("dMM linear fit:")
    slope_dMM = result.params[1]
    slope_std_err = result.bse[1]
    print("Slope: {:.2f} +/- {:.2f}".format(slope_dMM, slope_std_err))

    dMM_mis_max_vel = []
    for i in range(len(dMM_mot)):
        if dMM_mot[i] == np.max(dMM_mot):
            dMM_mis_max_vel.append(dMM_mis[i])
            
    dMM_mis_max_vel = np.array(dMM_mis_max_vel)
    print(f"Min at max vel: {np.min(dMM_mis_max_vel):.2f}")
    print(f"Max at max vel: {np.max(dMM_mis_max_vel):.2f}")
    print(f"Std at max vel: {np.std(dMM_mis_max_vel):.2f}")
    print("---------------------")
    
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
    plt.title('Population-Code Model: dMM Neurons', fontsize=16)
    plt.grid(True)

    fname = 'dMM_speed.eps'
    plt.savefig(fname, format='eps')
    plt.show()

    # Fit linear model to hMM data
    X = sm.add_constant(hMM_mot)
    model = sm.RLM(hMM_mis, X, M=sm.robust.norms.TukeyBiweight())
    result = model.fit()
    print("hMM linear fit:")
    slope_hMM = result.params[1]
    slope_std_err = result.bse[1]
    print("Slope: {:.2f} +/- {:.2f}".format(slope_hMM, slope_std_err))

    hMM_mis_max_vel = []
    for i in range(len(hMM_mot)):
        if hMM_mot[i] == np.max(hMM_mot):
            hMM_mis_max_vel.append(hMM_mis[i])

    count_pos_mis = 0
    for i in range(len(hMM_mis)):
        if hMM_mis[i] > 0:
            count_pos_mis += 1
            
    hMM_mis_max_vel = np.array(hMM_mis_max_vel)
    print(f"Min at max vel: {np.min(hMM_mis_max_vel):.2f}")
    print(f"Max at max vel: {np.max(hMM_mis_max_vel):.2f}")
    print(f"Std at max vel: {np.std(hMM_mis_max_vel):.2f}")
    
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
    plt.title('Population-Code Model: hMM Neurons', fontsize=16)
    plt.grid(True)
    
    fname = 'hMM_speed.eps'
    plt.savefig(fname, format='eps')
    plt.show()

    # Compute correlation coefficients between speed and mismatch response
    all_mot = np.zeros((N, num_trials * num_repeat))
    all_mis = np.zeros((N, num_trials * num_repeat))

    for m in range(num_trials):
        for i in range(N):
            for j in range(num_repeat):
                all_mot[i,m*num_repeat+j] = motor_flow[m]
                all_mis[i,m*num_repeat+j] = all_mismatch[m][i] + 0.15 * np.random.randn(1)
    
    correlation_coefficient = np.zeros(N)
    for i in range(N):
        x = all_mot[i,:]
        y = all_mis[i,:]
        # Compute the correlation coefficient
        correlation_matrix = np.corrcoef(x, y)
        correlation_coefficient[i] = correlation_matrix[0, 1]

    # Compute correlation coefficient for each neuron type
    cc_dMM = correlation_coefficient[np.unique(dMM_ind)]
    cc_hMM = correlation_coefficient[np.unique(hMM_ind)]
    # Create the set of all indices in the range(N)
    all_indices = set(range(N))
    un_ind = np.array(list(all_indices.difference(set(dMM_ind).union(set(hMM_ind)))))
    cc_un = correlation_coefficient[un_ind]

    # Plot the histogram
    plt.figure(figsize=(8, 5))
    plt.hist(correlation_coefficient, bins=20, color='gray', edgecolor='black', alpha=0.7)

    # Customize the plot
    plt.xlabel('Correlation (Locomotion Speed vs Mismatch Response)', fontsize=16)
    plt.ylabel('Neuron Count', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Remove box around plot but keep axes
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    
    fname = 'pop_code_correlation.eps'
    plt.savefig(fname, format='eps')
    plt.show()

    # Show the box plot of correlation coefficients for hMM, dMM, and unclass. neurons
    data = [cc_dMM, cc_un, cc_hMM]
    labels = ['dMM', 'Unclassified', 'hMM']

    # Create a figure with subplots
    fig, ax = plt.subplots(figsize=(8, 5))

    # Add vertical dashed line at x=0
    plt.axvline(x=0, linestyle='--', color='gray', linewidth=2)
    
    # Plot each vertical box plot with individual data points
    for i, d in enumerate(data):
        # Position the box plot vertically
        position = i + 1
    
        # Plot box plot
        sns.boxplot(x=d, y=np.ones_like(d)*position, orient='h', color='lightgray', whis=[0, 100], width=0.4, boxprops=dict(linewidth=2.5), medianprops=dict(linewidth=3.5), whiskerprops=dict(linewidth=2.5), capprops=dict(linewidth=2.5))
    
        # Plot strip plot of individual data points
        sns.stripplot(x=d, y=np.ones_like(d)*position, orient='h', color='black', alpha=1, jitter=True)

    plt.yticks(range(0, len(labels)), labels, fontsize=16)
    plt.xlabel('Correlation (Locomotion Speed vs Mismatch Response)', fontsize=16)
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Remove box around plot but keep axes
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    fname = 'correlation_boxplot.eps'
    plt.savefig(fname, format='eps')
    plt.show()

if __name__ == "__main__":
    main()
