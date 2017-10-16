import cv2
import numpy as np
import math

"""
Each variable x represents each pixel with RGB value, which is 3 dimensional vector.
To do foreground background segmentation, set number of cluster K = 2.
"""

def Initialization(K, height, width, channel):
    # pi is the mixing coefficient (prior probability). 
    # K dimensional vector. equal probable for every class.
    # pi's elements sum up to 1.
    pi = np.repeat(np.array([1.0/K]),K) 
    
    # u represtns mean  of the mixture Gaussian model. K dim 3d-array.
    # random initialization performs well for cow, fox, zebra
    # for the owl case: u = np.array([[239,244,245],[104,140,170]])
    u = 255.0*np.random.rand(K,channel) 
    
    # covmat is the variance of the mixture Gaussian model. K*channel*channel tensor.
    covmat = np.repeat(np.array([[[255.0,0.0,0.0],[0.0,255.0,0.0],[0.0,0.0,255.0]]]),K,axis=0)
    
    # The table of conditional distribution probability. K*height*width tensor.
    probmat = np.zeros([K,height,width])
    
    # The deviation of pixel value from the mean: x2 = x - u.
    # K*height*width*channel tensor.
    x2 = np.zeros([K,height,width,channel])
    
    return pi, u, covmat, probmat, x2


img = cv2.imread('fox.jpg')
(height,width,channel)=img.shape
print("Image shape is: ", img.shape)

#number of clusters
K = 2

#Initialization
pi, u, covmat, probmat, x2 = Initialization(K, height, width, channel)



"""
This cell runs the EM algorithm. Can be run multiple times to increase iterations.
"""
total_iteration = 20
for i in range(total_iteration):
    print("Iteration = ",i)
    
    #==========================E step. Update gamma===========================
    
    # Determinant of the covmat. K dimentional vector.
    covmatDet = np.linalg.det(covmat)
    
    # Inverse of the covmat. Used in calculation of normal probability.
    # K*channel*channel tensor
    covmatInv = np.linalg.inv(covmat)
    
    # ------- update x2 ----------
    for k in range(K):    
        x2[k] = img - u[k]
    
    # ------- update probmat --------
    probmat = np.einsum("a, abc -> abc", 
                        1/np.sqrt(2*math.pi*covmatDet), 
                        np.exp(-0.5* (x2*np.einsum("aed, abcd -> abce", covmatInv, x2)).sum(axis=3)))

    # ------- update gamma --------
    # completeProbmat is the posterior distribution probabililty. K*height*width tensor. 
    completeProbmat = np.einsum("a, abc -> abc", pi, probmat)
    completeProbmatSumRec = 1/completeProbmat.sum(axis=0) #denominator
    
    # Gamma is the responsibility.
    gamma = completeProbmat * completeProbmatSumRec

    
    #====================M step. Update pi, u, covmat===========================
    
    # ------- Update N and pi-------
    N = gamma.sum(axis=1).sum(axis=1)
    pi = N/(gamma.sum())
    # print("Value of pi = ",pi)
    
    # ------- Update u ---------
    u = np.einsum("a, ab -> ab", 1/N, np.einsum("abc, bcd -> ad", gamma, img))
    
    # -------- Update covmat -------
    for k in range(K):    
        x2[k] = img - u[k]
    x_mat = np.einsum("abcd, abce -> abcde", x2, x2)
    covmat = np.einsum("a, abc -> abc", 1/N, np.einsum("abc, abcde -> ade", gamma, x_mat))
    # --------- check the singularity of covmat --------
    for k in range(K):
        if np.linalg.det(covmat[k])<1.0:
            # print("Near exception!")
            covmat[k] = np.array([[10.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]])
    print("fraction = ",pi)



"""
Show filter.
"""

heatmap = np.zeros([height,width,3])
for h in range(height):
    for w in range(width):
        heatmap[h,w,:] = np.array(u[np.argmax(probmat[:,h,w])],dtype=float)

cv2.imwrite("fox_2.jpg",heatmap)
