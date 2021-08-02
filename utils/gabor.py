import numpy as np
import cv2

def get_gabor_kernel_bank(ksize, num_theta, lam):
    #################################################################################
    # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    # ksize - size of gabor filter (n, n)
    # sigma - standard deviation of the gaussian function
    # theta - orientation
    # lambda - wavelength of the sinusoidal factor
    # gamma - spatial aspect ratio
    # psi - phase offset
    # ktype - type and range of values. Can be cv2.CV_32F or cv2.CV_64F
    ################################################################################
    sigma = ksize / 5.0
    psi = 90 * np.pi / 180.0
    gamma = 1.5

    theta = np.zeros(num_theta)
    for k in range(num_theta):
        theta[k] = np.pi * k / num_theta

    ker_r = np.zeros([ksize, ksize, len(theta)*len(lam)])
    ker_i = np.zeros([ksize, ksize, len(theta)*len(lam)])

    k=0
    for lm in lam:
        for kk, th in zip(range(num_theta), theta):
            ker_r[:,:,k] = cv2.getGaborKernel((ksize,ksize), sigma, th, lm, gamma, 0)
            ker_i[:,:,k] = cv2.getGaborKernel((ksize,ksize), sigma, th, lm, gamma, psi)
            k=k+1
    return ker_r, ker_i


def apply_gabor(img, Gr, Gi):
    # This function returns a list that has as many elements as the number of orientations
    # of the Gabor kernel. The generic element 'i' is as a vector (the vector size is the
    # number of image pixels)that stores the normalized response for each image pixel to
    # Gabor kernels corresponding to the orientation 'i'
    pixel_features = []
    for k in range(Gr.shape[2]):
        response_r = cv2.filter2D(img, cv2.CV_32F, Gr[:, :, k])
        response_i = cv2.filter2D(img, cv2.CV_32F, Gi[:, :, k])
        response_mag = np.sqrt(response_r ** 2 + response_i ** 2)
        response_mean = 0.0
        response_std = 1.0
        response = (response_mag - response_mean) / response_std
        pixel_features.append(response.reshape(-1))
    return pixel_features
