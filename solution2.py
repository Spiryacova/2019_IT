import matplotlib.pyplot as plt
import numpy as np
from math import exp,pow, pi

def add_noise(img, rate=5):
    img[::rate, ::rate, :] = 1
    return

def gauss(dif, sigma):
    return exp(-2*pow((dif/sigma),2)/(sigma*pow(2*pi,1/2)))

def gauss_kernel(width, height, sigma):
    kernel = np.ones((width, height))
    for i in range(height):
        for j in range(width):
            dif = (i-(height-1)/2)**2 + (j-(width-1)/2)**2
            kernel[i,j] = gauss(dif, sigma)
    kernel/=kernel.sum()
    return kernel


def filter(img, window_size=3):
    img_gauss = np.zeros_like(img)
    kernel = gauss_kernel(window_size, window_size,10)
    p = window_size//2
    for i in range(window_size // 2, img.shape[0] - window_size // 2):
        for j in range(window_size // 2, img.shape[1] - window_size // 2):
            for k in range(img.shape[2]):
                conv = img[i - p: i + p + 1, j - p: j + p + 1, k] * kernel
                img_gauss[i, j, k] = conv.sum()
    return img_gauss

img_raw = plt.imread("img.jpg")[:, :, :3]
img_gauss = filter(img_raw, gauss_kernel(7, 10))

fig, axs = plt.subplots(1,2)
axs[0].imshow(img_raw)
axs[1].imshow(img_gauss)
img_gauss.show()