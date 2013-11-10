import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.signal import decimate

from pycoasters.coaster import Coaster, a2color
from pycoasters.images import show

def pdf2png(fname):
    """
    Matplotlib makes beautiful PDFs, but some ugly PNGs of my polar plots. sips does a better job
    """
    try:
        subprocess.call(['sips', '-s', 'format', 'png', '--resampleWidth', '1000', fname, '--out', fname.replace('.pdf', '.png')])
    except OSError:
        # Fall back to imageMagick, if it exists
        subprocess.call(['convert', fname, fname.replace('.pdf', '.png')])

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
        fname = os.path.join(ride_folder, 'accel_3d.pdf')
        plt.savefig(fname)
        pdf2png(fname)

        # Make 2D plots
        f, axs = plt.subplots(3,1,figsize=(8,16))
        c.plot_magnitude(axs[0])
        c.plot_original_xyz(axs[1])
        c.plot_reoriented_xyz(axs[2])
        fname = os.path.join(ride_folder, 'acceleration.pdf')
        plt.savefig(fname)
        pdf2png(fname)

        # Make portraits
        img = c.portrait_square()
        mag = 10
        img = np.dstack([np.kron(img[:,:,j], np.ones((mag,mag))) for j in range(img.shape[2])])
        fname = os.path.join(ride_folder, 'accel_portrait.pdf')
        show(img, fname)
        pdf2png(fname)

        img = c.portrait_line()
        fname = os.path.join(ride_folder, 'accel_portrait_line.pdf')
        show(img, fname)
        pdf2png(fname)

        if 'skip' in c.notes and c.notes['skip']:
            continue

        # Make polar portrait
        plt.figure(figsize=(10,10))
        ax = plt.subplot(111,polar=True)
        c.plot_portrait_polar(ax)
        fname = os.path.join(ride_folder, 'accel_polar.pdf')
        plt.savefig(fname)
        pdf2png(fname)

if __name__ == '__main__':
    main()
