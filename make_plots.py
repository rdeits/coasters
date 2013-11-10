import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.signal import decimate

from pycoasters.coaster import Coaster, a2color
from pycoasters.images import show

def main():
    base_path = sys.argv[1]

    folders = []

    def visit(arg, dirname, names):
        if 'accelerometer_log.txt' in names:
            folders.append(dirname)
    os.path.walk(base_path, visit, None)

    for ride_folder in folders:
        print "Running analysis on {}".format(ride_folder)
        c = Coaster.load(ride_folder)

        # Make 3D plot
        fig = plt.figure(figsize=(11,11))
        ax = fig.add_subplot(111, projection='3d')
        c.plot_xtz_3d(ax)
        ax.view_init(15,330)
        plt.savefig(os.path.join(ride_folder, 'accel_3d.png'))
        plt.savefig(os.path.join(ride_folder, 'accel_3d.pdf'))

        # Make 2D plots
        f, axs = plt.subplots(3,1,figsize=(8,16))
        c.plot_magnitude(axs[0])
        c.plot_original_xyz(axs[1])
        c.plot_reoriented_xyz(axs[2])
        plt.savefig(os.path.join(ride_folder, 'acceleration.pdf'))
        plt.savefig(os.path.join(ride_folder, 'acceleration.png'))

        # Make portraits
        img = c.portrait_square()
        mag = 10
        img = np.dstack([np.kron(img[:,:,j], np.ones((mag,mag))) for j in range(img.shape[2])])
        show(img, os.path.join(ride_folder, 'accel_portrait.pdf'))

        img = c.portrait_line()
        show(img, os.path.join(ride_folder, 'accel_portrait_line.pdf'))

        if 'skip' in c.notes and c.notes['skip']:
            continue

        # Make polar portrait
        plt.figure(figsize=(25,25))
        ax = plt.subplot(111,polar=True)
        c.plot_portrait_polar(ax)
        plt.savefig(os.path.join(ride_folder, 'accel_polar.png'))
        plt.savefig(os.path.join(ride_folder, 'accel_polar.pdf'))
if __name__ == '__main__':
    main()
