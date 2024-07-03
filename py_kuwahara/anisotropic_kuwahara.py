# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:39:00 2018

@author: raymondmg
"""

import numpy as np
import cv2 as cv
import SST
import sys
from tqdm import tqdm

class AnisotropicKuwahara:
    def __init__(self, image, sst, kernel_size=7, div_num=8, q=8.0, alpha=1.0):
        self.sst = sst
        self.kernel_size = kernel_size
        self.image = image
        self.div_num = div_num
        self.angle = 2 * np.pi / div_num
        self.q = q
        self.alpha = alpha
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.channel = image.shape[2]
        self.param_anisotropy()

    def param_anisotropy(self):
        visual_image = cv.imread("./img/visual_rgb.png")
        anisotropic_image = np.zeros((self.height, self.width, self.channel))
        A = np.zeros((self.height, self.width))
        PHI = np.zeros((self.height, self.width))

        # Adding progress tracking with tqdm
        for j in tqdm(range(self.height), desc="Processing Rows"):
            for i in range(self.width):
                E = self.sst[j, i, 0]
                G = self.sst[j, i, 1]
                F = self.sst[j, i, 2]
                D = np.sqrt((E - G) * (E - G) + 4.0 * F * F)

                lambda1 = (E + G + D) / 2.0
                lambda2 = (E + G - D) / 2.0

                if (lambda1 + lambda2) <= 0:
                    A[j, i] = 0
                else:
                    A[j, i] = (lambda1 - lambda2) / (lambda1 + lambda2)

                # visualization Anisotropic
                anisotropic_image[j, i, 0] = visual_image[0, int(255 * A[j, i]), 0]
                anisotropic_image[j, i, 1] = visual_image[0, int(255 * A[j, i]), 1]
                anisotropic_image[j, i, 2] = visual_image[0, int(255 * A[j, i]), 2]

                PHI[j, i] = np.arctan2(-F, lambda1 - E)

        self.A = A
        self.PHI = PHI
        cv.imwrite("./img/anisotropic_image.png", anisotropic_image)
        return anisotropic_image

def main(image_path):
    img = cv.imread(image_path)
    if img is None:
        print("Error: Unable to read image. Please check the path.")
        return
    
    sst_func = SST.SST(img, SST.SST_TYPE.CLASSIC)
    sst_image = sst_func.cal(5)

    aniso_kuwahara_func = AnisotropicKuwahara(img, sst_image)
    anisotropic_image = aniso_kuwahara_func.param_anisotropy()

    output_path = "./img/anisotropic_image.png"
    cv.imwrite(output_path, anisotropic_image)
    print(f"Anisotropic Kuwahara filtered image saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python anisotropic_kuwahara.py <image_path>")
    else:
        main(sys.argv[1])
