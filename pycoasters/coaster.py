from __future__ import division, print_function

import os.path
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.signal import resample

from pycoasters.rotations import axis2rotmat

g = 9.81
TARGET_SAMPLE_HZ = 30
DEFAULT_NOTES = {"ride_start": 0, "ride_end": -1, "header_lines": 1, "footer_lines": 0,
                 "acceleration_units_per_g": 1, "time_units_per_s": 1}

def mean_unit_vector(data, tspan):
    chunk = data[((tspan[0] <= data.time) & (data.time <= tspan[1]))]
    z = chunk.z.mean()
    y = chunk.y.mean()
    x = chunk.x.mean()
    v = np.array([x,y,z])
    v = v / norm(v)
    return v


def a2color(x,y,z):
    color = np.zeros((len(x),3),dtype='i8')
    m = 255.0 / 6.5
    b = 255.0 * 2 / 6.5
    for j, v in enumerate((x,y,z-1)):
        color[:,j] = v * m + b
        color[color[:,j] > 255,j] = 255
        color[color[:,j] < 0, j] = 0
    return color


class Coaster(object):
    @staticmethod
    def load(ride_folder):
        notes_fn = os.path.join(ride_folder, 'notes.json')
        try:
            loaded_notes = json.load(open(notes_fn, 'r'))
        except IOError:
            loaded_notes = {}
        notes = DEFAULT_NOTES.copy()
        notes.update(loaded_notes)
        header = notes["header_lines"]
        footer = notes["footer_lines"]

        data = pd.read_csv(os.path.join(ride_folder, 'accelerometer_log.txt'),
                           sep=",", header=header, skip_footer=footer)
        return Coaster(data, notes)

    def __init__(self, data, notes):
        self.data = data
        self.notes = notes
        self.data.time = self.data.time / notes["time_units_per_s"]
        self.data.x = self.data.x / self.notes["acceleration_units_per_g"]
        self.data.y = self.data.y / self.notes["acceleration_units_per_g"]
        self.data.z = self.data.z / self.notes["acceleration_units_per_g"]

        tfi = self.data.time.last_valid_index()
        timespan = self.data.time[tfi] - self.data.time[0]
        sample_rate = len(self.data.time) / timespan

        if sample_rate > TARGET_SAMPLE_HZ:
            resampled_data = {}
            for k in self.data.keys():
                if k in ['x', 'y', 'z']:
                    (resampled_data[k], resampled_data['time']) = \
                        resample(self.data[k][:tfi],
                                 t=self.data.time[:tfi],
                                 num=timespan*TARGET_SAMPLE_HZ)
            resampled_data['time'] = resampled_data['time'][:len(resampled_data['x'])]
            self.data = pd.DataFrame.from_dict(resampled_data)

        self.ndx_range = (notes['ride_start'], notes['ride_end'])

        self.tspan_z = None
        self.tspan_zx = None
        if 'known_orientations' in notes:
            if 'zhat' in notes['known_orientations']:
                self.tspan_z = notes['known_orientations']['zhat']
            if 'z-x' in notes['known_orientations']:
                self.tspan_zx = notes['known_orientations']['z-x']

        self.r_data = self.reorient()
        self.x = np.array(self.r_data.x[self.ndx_range[0]:self.ndx_range[1]])
        self.y = np.array(self.r_data.y[self.ndx_range[0]:self.ndx_range[1]])
        self.z = np.array(self.r_data.z[self.ndx_range[0]:self.ndx_range[1]])
        self.t = np.array(self.r_data.time[self.ndx_range[0]:self.ndx_range[1]])

    def reorient(self):
        if self.tspan_z is not None:
            zhat_0 = mean_unit_vector(self.data, self.tspan_z)
            axis = np.cross(np.array([0,0,1]), zhat_0)
            angle = -np.arctan2(norm(axis), np.dot(np.array([0,0,1]), zhat_0))
            R = axis2rotmat(np.hstack((axis, angle)))
            rotated_data = np.dot(R, np.vstack([np.transpose(self.data.x), np.transpose(self.data.y), np.transpose(self.data.z)]))

            if self.tspan_zx is not None:
                zx_data = rotated_data[:,(self.tspan_zx[0] <= self.data.time) & (self.data.time <= self.tspan_zx[1])]
                x = np.mean(zx_data[0,:])
                y = np.mean(zx_data[1,:])
                z = np.mean(zx_data[2,:])
                zx = np.array([x,y,z])
                zx = zx / norm(zx)
                axis = np.array([0,0,1])
                angle = -np.arctan2(zx[1], zx[0])
                R = axis2rotmat(np.hstack((axis, angle)))
                rotated_data = np.dot(R, rotated_data)

            r_data = self.data.copy()
            r_data.x = rotated_data[0,:]
            r_data.y = rotated_data[1,:]
            r_data.z = rotated_data[2,:]
        else:
            r_data = self.data
        return r_data

    def plot_magnitude(self, ax):
        h = ax.plot(self.t, np.sqrt(np.power(self.x, 2) + np.power(self.y, 2) + np.power(self.z, 2)))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (g)')
        ax.set_ylim([-1, 5])
        if 'title' in self.notes:
            ax.set_title(self.notes['title'])
        return h

    def plot_original_xyz(self, ax):
        start, end = self.ndx_range
        ax.plot(self.t, self.data.x[start:end], 'r-', self.t, self.data.y[start:end], 'g-', self.t, self.data.z[start:end], 'b-')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Acceleration (g)')
        ax.legend(['x', 'y', 'z'])
        if self.tspan_z is not None:
            ax.axvspan(self.tspan_z[0],self.tspan_z[1],facecolor='b',alpha=0.2)
        if self.tspan_zx is not None:
            ax.axvspan(self.tspan_zx[0],self.tspan_zx[1],facecolor='r',alpha=0.2)
        if 'title' in self.notes:
            ax.set_title(self.notes['title'])

    def plot_reoriented_xyz(self, ax):
        ax.plot(self.t, self.x, 'r-', self.t, self.y, 'g-', self.t, self.z, 'b-')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Acceleration (g)')
        ax.legend(['x', 'y', 'z'])
        if self.tspan_z is not None:
            ax.axvspan(self.tspan_z[0],self.tspan_z[1],facecolor='b',alpha=0.2)
        if self.tspan_zx is not None:
            ax.axvspan(self.tspan_zx[0],self.tspan_zx[1],facecolor='r',alpha=0.2)
        if 'title' in self.notes:
            ax.set_title(self.notes['title'])

    def accel_colors(self, scale=1):
        """
        Convert acceleration data into pretty colors
        """

        tup = (self.x * scale, self.z * scale, self.y * scale)  # Using (x,z,y) -> (red,green,blue) map for aesthetics of final result
        return a2color(*tup)

    def portrait_square(self):
        img = self.accel_colors()
        d = np.ceil(np.sqrt(len(self.t)))
        img.resize((d**2,3))
        img = img.reshape((d,d,3))
        return img

    def portrait_line(self):
        img = self.accel_colors()
        d = len(self.t)
        img = img.reshape((1,d,3))
        img = np.resize(img, (500, d, 3))
        return img

    def plot_portrait_polar(self, ax):
        s = 8  # arbitrary scaling factor
        mag = np.sqrt(np.power(self.x, 2) + np.power(self.y, 2) + np.power(self.z, 2))
        img = self.accel_colors(scale=s/mag)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2)

        ax.bar(np.linspace(0,2*np.pi,img.shape[0]),
               np.log(mag+1),
               width=2*np.pi/(img.shape[0])*0.9,
               color=img[:,[0,2,1]]/255.0,
               linewidth=0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_ylim([0, np.log(5)])

    def plot_xtz_3d(self, ax, cmap=plt.get_cmap('copper')):
        ax.hold(True)
        for (i, ti) in enumerate(self.t):
            c = cmap((ti-self.t[0])/(self.t[-1]-self.t[0]))
            ax.plot3D([self.x[i]], [ti], zs=[self.z[i]],
                      color=c,
                      markeredgecolor=[ci/2 for ci in c],
                      linestyle='',
                      marker='.')
        ax.set_xlabel('x (g)')
        ax.set_ylabel('time (s)')
        ax.set_zlabel('z (g)')
        ax.view_init(15,330)
        if 'title' in self.notes:
            ax.set_title(self.notes['title'])

