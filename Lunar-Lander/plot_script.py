import numpy as np
import matplotlib.pyplot as plt
import pickle

plt_legend_dict = {"expected_sarsa_agent": "Expected SARSA with neural network",
                   "random_agent": "Random"}
path_dict = {"expected_sarsa_agent": "results/",
             "random_agent": "./"}

plt_label_dict = {"expected_sarsa_agent": "Sum of\nreward\nduring\nepisode"}

# Plot neural networks
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], lw = 0.1, c='k')
                ax.add_artist(line)

def smooth(data, k):
    num_episodes = data.shape[1]
    num_runs = data.shape[0]

    smoothed_data = np.zeros((num_runs, num_episodes))

    for i in range(num_episodes):
        if i < k:
            smoothed_data[:, i] = np.mean(data[:, :i+1], axis = 1)   
        else:
            smoothed_data[:, i] = np.mean(data[:, i-k:i+1], axis = 1)    
        

    return smoothed_data

# Function to plot result
def plot_result(data_name_array):
    plt_agent_sweeps = []
    
    fig, ax = plt.subplots(figsize=(8,6))

    
    for data_name in data_name_array:
        
        # load data
        filename = 'sum_reward_{}'.format(data_name).replace('.','')
        sum_reward_data = np.load('{}/{}.npy'.format(path_dict[data_name], filename))

        # smooth data
        smoothed_sum_reward = smooth(data = sum_reward_data, k = 100)
        
        mean_smoothed_sum_reward = np.mean(smoothed_sum_reward, axis = 0)

        plot_x_range = np.arange(0, mean_smoothed_sum_reward.shape[0])
        graph_current_agent_sum_reward, = ax.plot(plot_x_range, mean_smoothed_sum_reward[:], label=plt_legend_dict[data_name])
        plt_agent_sweeps.append(graph_current_agent_sum_reward)
    
    ax.legend(handles=plt_agent_sweeps, fontsize = 13)
    ax.set_title("Learning Curve", fontsize = 15)
    ax.set_xlabel('Episodes', fontsize = 14)
    ax.set_ylabel(plt_label_dict[data_name_array[0]], rotation=0, labelpad=40, fontsize = 14)
    ax.set_ylim([-300, 300])

    plt.tight_layout()
    plt.show()     
