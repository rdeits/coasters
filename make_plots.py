import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

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
        plt.savefig(os.path.join(ride_folder, 'accel_3d.pdf'))
        plt.savefig(os.path.join(ride_folder, 'accel_3d.svg'))

        # Make 2D plots
        f, axs = plt.subplots(3,1,figsize=(8,16))
        c.plot_magnitude(axs[0])
        c.plot_original_xyz(axs[1])
        c.plot_reoriented_xyz(axs[2])
        plt.savefig(os.path.join(ride_folder, 'acceleration.svg'))
        plt.savefig(os.path.join(ride_folder, 'acceleration.pdf'))
        plt.savefig(os.path.join(ride_folder, 'acceleration.png'))

        # Make portraits
        img = c.portrait_square()
        mag = 10
        img = np.dstack([np.kron(img[:,:,j], np.ones((mag,mag))) for j in range(img.shape[2])])
        show(img, os.path.join(ride_folder, 'accel_portrait.pdf'))

        img = c.portrait_line()
        show(img, os.path.join(ride_folder, 'accel_portrait_line.pdf'))

        # Make polar portrait
        mag = np.sqrt(np.power(c.x, 2) + np.power(c.y, 2) + np.power(c.z, 2))
        s = 8
        tup = (c.x * s/mag, c.z * s/mag, c.y * s/mag)  # order chosen for aesthetics
        img = a2color(*tup)
        plt.figure(figsize=(25,25))
        ax = plt.subplot(111,polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2)
        # ax.plot([0,0], [np.log(mag[0]+1), np.log(5)], '--', color=[0.5,0.5,0.5,0.5], linewidth=5)
        p = ax.bar(np.linspace(0,2*np.pi,img.shape[0]),
                   np.log(mag+1),
                   width=2*np.pi/img.shape[0],
                   color=img[:,[0,2,1]]/255.0,
                   linewidth=0)
        # ax.set_ylim([0,4])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_ylim([0, np.log(5)])
        plt.rc('text', usetex=True)
        plt.rc('font', family='sans-serif')
        ax.set_title(c.notes['title'],
                     fontdict={'size':58})
        plt.savefig(os.path.join(ride_folder, 'accel_polar.png'))
if __name__ == '__main__':
    main()
