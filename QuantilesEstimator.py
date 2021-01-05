import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy as sp
from scipy.special import gamma as g

from scipy.stats import levy_stable as levy

import matplotlib.patches as mpatches

table_3 = np.array([[2.   , 2.   , 2.   , 2.   , 2.   , 2.   , 2.   ],
       [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
       [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
       [1.729, 1.73 , 1.737, 1.745, 1.745, 1.745, 1.745],
       [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
       [1.563, 1.56 , 1.553, 1.548, 1.547, 1.547, 1.547],
       [1.484, 1.48 , 1.471, 1.46 , 1.448, 1.438, 1.438],
       [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
       [1.279, 1.273, 1.266, 1.25 , 1.21 , 1.184, 1.15 ],
       [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
       [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
       [0.896, 0.892, 0.887, 0.883, 0.855, 0.823, 0.769],
       [0.818, 0.812, 0.806, 0.801, 0.78 , 0.756, 0.691],
       [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.595],
       [0.593, 0.59 , 0.588, 0.586, 0.579, 0.563, 0.513]])


table_4 = np.array([[0.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   ],
       [0.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   ],
       [0.   , 0.759, 1.   , 1.   , 1.   , 1.   , 1.   ],
       [0.   , 0.482, 1.   , 1.   , 1.   , 1.   , 1.   ],
       [0.   , 0.36 , 0.76 , 1.   , 1.   , 1.   , 1.   ],
       [0.   , 0.253, 0.518, 0.823, 1.   , 1.   , 1.   ],
       [0.   , 0.203, 0.41 , 0.632, 1.   , 1.   , 1.   ],
       [0.   , 0.165, 0.332, 0.499, 0.943, 1.   , 1.   ],
       [0.   , 0.136, 0.271, 0.404, 0.689, 1.   , 1.   ],
       [0.   , 0.109, 0.216, 0.323, 0.539, 0.827, 1.   ],
       [0.   , 0.096, 0.19 , 0.284, 0.472, 0.693, 1.   ],
       [0.   , 0.082, 0.163, 0.243, 0.412, 0.601, 1.   ],
       [0.   , 0.074, 0.147, 0.22 , 0.377, 0.546, 1.   ],
       [0.   , 0.064, 0.128, 0.191, 0.33 , 0.478, 0.362],
       [0.   , 0.056, 0.112, 0.167, 0.285, 0.428, 1.   ]])


table_5 = np.array([[1.908, 1.908, 1.908, 1.908, 1.908],
       [1.914, 1.915, 1.916, 1.918, 1.921],
       [1.921, 1.922, 1.927, 1.936, 1.947],
       [1.927, 1.93 , 1.943, 1.961, 1.987],
       [1.933, 1.94 , 1.962, 1.997, 2.043],
       [1.939, 1.952, 1.988, 2.045, 2.116],
       [1.946, 1.967, 2.022, 2.106, 2.211],
       [1.955, 1.984, 2.067, 2.188, 2.333],
       [1.965, 2.007, 2.125, 2.294, 2.491],
       [1.98 , 2.04 , 2.205, 2.435, 2.696],
       [2.   , 2.085, 2.311, 2.624, 2.973],
       [2.04 , 2.149, 2.461, 2.886, 3.356],
       [2.098, 2.244, 2.676, 3.265, 3.912],
       [2.189, 2.392, 3.004, 3.844, 4.775],
       [2.336, 2.635, 3.542, 4.808, 6.247],
       [2.588, 3.073, 4.534, 6.636, 9.144]])

table_6 = np.array([[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   , -0.017, -0.032, -0.049, -0.064],
       [ 0.   , -0.03 , -0.061, -0.092, -0.123],
       [ 0.   , -0.043, -0.088, -0.132, -0.179],
       [ 0.   , -0.056, -0.111, -0.17 , -0.232],
       [ 0.   , -0.066, -0.134, -0.206, -0.283],
       [ 0.   , -0.075, -0.154, -0.241, -0.335],
       [ 0.   , -0.084, -0.173, -0.276, -0.39 ],
       [ 0.   , -0.09 , -0.192, -0.31 , -0.447],
       [ 0.   , -0.095, -0.208, -0.346, -0.508],
       [ 0.   , -0.098, -0.223, -0.383, -0.576],
       [ 0.   , -0.099, -0.237, -0.424, -0.652],
       [ 0.   , -0.096, -0.25 , -0.469, -0.742],
       [ 0.   , -0.089, -0.262, -0.52 , -0.853],
       [ 0.   , -0.078, -0.272, -0.581, -0.997],
       [ 0.   , -0.061, -0.279, -0.659, -1.198]])


v_beta_values = [0.0,0.1,0.2,0.3,0.5,0.7,1.0]

v_alpha_values = [2.439,2.5,2.6,2.7,2.8,3.0,3.2,3.5,4.0,5.0,6.0,8.0,10.0,15.0,25.0]

alpha_values = [2.0,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1,0.9,0.8,0.7,0.6,0.5]

beta_values = [0.0,0.25,0.50,0.75,1.0]

def Quantile(samples):
    #s is an array of samples

    #quantiles for alpha and beta
    v_alpha = (np.quantile(samples,0.95) - np.quantile(samples,0.05))/(np.quantile(samples,0.75) - np.quantile(samples,0.25))
    v_beta = (np.quantile(samples,0.05) + np.quantile(samples,0.95) - 2*np.quantile(samples,0.5))/(np.quantile(samples,0.95)- np.quantile(samples,0.05))

    #search the tables
    v_beta_idx = 0
    for i in range(len(v_beta_values)):
        if v_beta_values[i+1] == np.abs(np.around(v_beta,1)):
            v_beta_idx = i+1
            break
        elif v_beta_values[i] <= np.abs(np.around(v_beta,1)) <= v_beta_values[i+1]:
            v_beta_idx = i
            break


    v_alpha_idx = 0
    for j in range(len(v_alpha_values)):
        if v_alpha_values[j+1] == np.around(v_alpha,1):
            v_alpha_idx = j+1
            break
        elif v_alpha_values[j] <= np.around(v_alpha,1) < v_alpha_values[j+1]:
            v_alpha_idx = j
            break

    #compute estimate for alpha
    alpha = table_3[v_alpha_idx][v_beta_idx]

    #compute estimate for beta
    if v_beta < 0:
        beta = -table_4[v_alpha_idx][v_beta_idx]
    else:
        beta = table_4[v_alpha_idx][v_beta_idx]

    #search the tables again
    beta_idx1 = 0
    for i in range(len(beta_values)):
        if beta_values[i-1] <= np.abs(np.around(beta,1))<= beta_values[i]:
            beta_idx1 = i
            break

    alpha_idx1 = 0
    for j in range(len(alpha_values)):
        if alpha_values[j+1] <= np.around(alpha,1) <= alpha_values[j]:
            alpha_idx1 = j+1
            break


    #compute estimate of gamma
    gamma = (np.quantile(samples,0.75) - np.quantile(samples,0.25))/table_5[alpha_idx1][beta_idx1]

    #computes estimate of delta
    if beta < 0:
        delta = (np.quantile(samples,0.5) - gamma*table_6[alpha_idx1][beta_idx1]) - beta*gamma*np.tan((np.pi*alpha)/2)
    else:
        delta = (np.quantile(samples,0.5) + gamma*table_6[alpha_idx1][beta_idx1]) - beta*gamma*np.tan((np.pi*alpha)/2)


    print("\n","alpha_hat =",alpha,"\n","beta_hat =",beta,"\n","gamma_hat =",gamma,"\n","delta_hat =",delta)

    delta_moments = np.mean(samples)

    return alpha, beta, gamma, delta, delta_moments
