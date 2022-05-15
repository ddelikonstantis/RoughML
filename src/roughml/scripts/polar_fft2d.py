import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
import argparse
import os
import sys
from scipy.stats import spearmanr


def polar_FFT2d(*args):
    polar_fft_total = []
    for iter, arg in enumerate(args):
        # establish image alpha value
        # if alpha values change, expand list
        alpha, alpha_val = None, ["0.5","0.6","0.7","0.8","0.9","1.0"]
        for a in alpha_val:
            if arg.find(str(a)) != -1:
                # get image alpha value
                alpha = a
                break
        if alpha == None:
            print("Alpha cannot be established in loaded image")
            sys.exit()
        # read image as grayscale
        img = cv2.imread(arg, 0)
        # get image dimension
        rows, clmns = img.shape[0], img.shape[1]
        # get image FFT2d and shift sums to the center
        fft2d = np.fft.fftshift(np.fft.fft2(img))
        # get absolute fourier values and flatten array
        flat_fft2d = abs(fft2d.ravel(order='F'))
        # get image center points
        mid_idxrow, mid_idxcol = math.floor(rows / 2), math.floor(clmns / 2)
        # get FFT2d mean absolute column values
        fft2d_mean_col = np.mean(abs(fft2d), axis=0)
        # keep only second half of the mean FFT2d matrices
        fft2d_mean_col_half = fft2d_mean_col[mid_idxcol:clmns]
        # get FFT2d mean absolute row values
        fft2d_mean_row = np.mean(abs(fft2d), axis=1)
        # keep only second half of the mean FFT2d matrices
        fft2d_mean_row_half = fft2d_mean_row[mid_idxrow:rows]
        dist = np.zeros(shape=(rows, clmns), dtype='float')
        # get indexes of the middle element of a matrix
        for i in range(rows):
                for j in range(clmns):
                    dist[i][j] = math.sqrt(pow((i-mid_idxrow), 2) + pow((j-mid_idxcol), 2))
        # convert distance matrix as a column matrix
        flat_dist = dist.ravel(order='F')
        # construct 2d array (first column: distances from center point, second column: corresponding fft values)
        dist_fft = np.vstack((flat_dist, flat_fft2d)).T
        # Sort 2d array based on 1st column
        dist_fft_sorted = dist_fft[dist_fft[:,0].argsort()]
        # get polar FFT2d for surface
        minVal, maxVal = float(0.0), float(1.0)
        # get loop cycle depending on lowest dimension
        if rows < clmns:
            cycle_radius = math.floor(rows / 2)
        else:
            cycle_radius = math.floor(clmns / 2)
        polar_fft = np.zeros(shape=(cycle_radius, 2), dtype='float')
        step = int(1)
        for j in range(0, cycle_radius, step):
            sum_dis = float(0.0)
            sum_fft = float(0.0)
            count = int(0)
            # get radial sums of distance and FFT2d
            for i in range(dist_fft.shape[0]):
                if (dist_fft_sorted[i][0] > minVal) and (dist_fft_sorted[i][0] <= maxVal):
                    sum_dis= sum_dis + dist_fft_sorted[i][0]
                    sum_fft= sum_fft + dist_fft_sorted[i][1]
                    count = count + 1
            polar_fft[j][0] = sum_dis / count
            polar_fft[j][1] = sum_fft / count
            minVal = maxVal
            maxVal = maxVal + step

        # prepare directory
        path = os.path.join(os.path.dirname( __file__ ), 'output/', alpha + '/').replace('\\', '/')
        try:
            os.mkdir(path)
        except FileExistsError:
            # directory already exists
            pass

        # create filename
        filename = "image_" + str(iter)
        # plot FFT2d
        plt.figure()
        plt.xlabel('Spatial frequency (nm^{-1})')
        plt.ylabel('Fourier amplitude (nm^{-1})')
        plt.xscale('log'), plt.yscale('log')
        plt.title('FFT2D vectors')
        # plot polar FFT2d
        plt.plot(polar_fft[:,0], polar_fft[:,1], label = "Polar FFT2D")
        # plot mean rows FFT2d
        plt.plot(polar_fft[:,0], fft2d_mean_row_half, label = 'Mean rows FFT2D')
        # plot mean columns FFT2d
        plt.plot(polar_fft[:,0], fft2d_mean_col_half, label = 'Mean columns FFT2D')
        plt.legend()
        plt.savefig(path + filename)

        # append polar FFT2d values for current image
        polar_fft_total.append(np.array(polar_fft[:,1]))
    
    # get average polar FFT2d values for all images
    polar_fft_total_mean = np.mean(polar_fft_total, axis=0)
    # polar_fft_total_mean = np.column_stack((np.full(shape=polar_fft_total_mean.shape, fill_value=float(alpha), order='F'), polar_fft_total_mean))
    polar_fft_total_mean = np.column_stack((polar_fft[:,0], polar_fft_total_mean))
    # create csv file with mean vector values
    df = pd.DataFrame(polar_fft_total_mean, columns = ['Alpha', 'Polar FFT2d values'])
    df_filename = '/polar_fft2d_' + alpha + '.csv'
    df.to_csv(path + df_filename, index=None)
    
    return None



if __name__ == "__main__":
    # argument parser
    # parser = argparse.ArgumentParser(description = 'Image polar fft2d')
    # parser.add_argument('image1', help = 'directory of first image to compare')
    # parser.add_argument('image2', help = 'directory of second image to compare')
    # # parser.add_argument('image3', help = 'directory of third image to compare')
    # # parser.add_argument('image4', help = 'directory of fourth image to compare')
    # # parser.add_argument('image5', help = 'directory of fifth image to compare')
    # args = parser.parse_args()

    arg1=r"Output\Alpha effect\Standard\dataset_1000_128_03_00_03_04_04_0.90\CNNGenerator_CNNDiscriminator\2022_04_17_14_29_03_199460\Plots\grayscale\fake_00.png"
    arg2=r"Output\Alpha effect\Standard\dataset_1000_128_03_00_03_04_04_0.90\CNNGenerator_CNNDiscriminator\2022_04_17_14_29_03_199460\Plots\grayscale\fake_01.png"
    arg3=r"Output\Alpha effect\Standard\dataset_1000_128_03_00_03_04_04_0.90\CNNGenerator_CNNDiscriminator\2022_04_17_14_29_03_199460\Plots\grayscale\fake_02.png"
    arg4=r"Output\Alpha effect\Standard\dataset_1000_128_03_00_03_04_04_0.90\CNNGenerator_CNNDiscriminator\2022_04_17_14_29_03_199460\Plots\grayscale\fake_03.png"
    arg5=r"Output\Alpha effect\Standard\dataset_1000_128_03_00_03_04_04_0.90\CNNGenerator_CNNDiscriminator\2022_04_17_14_29_03_199460\Plots\grayscale\fake_04.png"
    arg6=r"Output\Alpha effect\Standard\dataset_1000_128_03_00_03_04_04_0.90\CNNGenerator_CNNDiscriminator\2022_04_17_14_29_03_199460\Plots\grayscale\fake_05.png"

    polar_FFT2d(arg1, arg2, arg3, arg4, arg5, arg6)

    # df1 = pd.read_csv(r'src\roughml\scripts\output\Alpha effect new\0.5\polar_fft2d_0.5.csv', header=0)
    # df1 = np.array(df1)
    # df2 = pd.read_csv(r'src\roughml\scripts\output\Alpha effect new\0.6\polar_fft2d_0.6.csv', header=0)
    # df2 = np.array(df2)
    # df3 = pd.read_csv(r'src\roughml\scripts\output\Alpha effect new\0.7\polar_fft2d_0.7.csv', header=0)
    # df3 = np.array(df3)
    # df4 = pd.read_csv(r'src\roughml\scripts\output\Alpha effect new\0.8\polar_fft2d_0.8.csv', header=0)
    # df4 = np.array(df4)
    # df5 = pd.read_csv(r'src\roughml\scripts\output\Alpha effect new\0.9\polar_fft2d_0.9.csv', header=0)
    # df5 = np.array(df5)
    # df6 = pd.read_csv(r'src\roughml\scripts\output\Alpha effect new\1.0\polar_fft2d_1.0.csv', header=0)
    # df6 = np.array(df6)
    

    # # df_mean_total = []
    # # df_mean_total = np.concatenate((df1, df2, df3, df4, df5, df6))

    # df_mean_total = df6

    # path = os.path.join(os.path.dirname( __file__ ), 'output/').replace('\\', '/')
    # try:
    #     os.mkdir(path)
    # except FileExistsError:
    #     # directory already exists
    #     pass
    
    # df_total = pd.DataFrame(df_mean_total)
    # df_filename = '/polar_fft2d_total.csv'
    # df_total.to_csv(path + df_filename, index=None, header=None)

    # coef, p = spearmanr(df_mean_total)
    # # print('Spearmans coefficient: ', coef, '\n', 'p-value: ', p)
    # spr = np.array([[coef, p]])
    # df_spr = pd.DataFrame(spr, columns = ['Spearmans coefficient', 'p-value'])
    # df_spr_filename = '/Spearmans rank correlation coefficient.csv'
    # df_spr.to_csv(path + df_spr_filename, index=None)

