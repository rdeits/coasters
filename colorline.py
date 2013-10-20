# Topics: line, color, LineCollection, cmap, colorline, codex
'''
Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
The color is taken from optional data in z, and creates a LineCollection.

z can be:
- empty, in which case a default coloring will be used based on the position along the input arrays
- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
- an array of the length of at least the same length as x, to color according to this data
- an array of a smaller length, in which case the colors are repeated along the curve

The function colorline returns the LineCollection created, which can be modified afterwards.

See also: plt.streamplot
'''

'''
source: http://nbviewer.ipython.org/urls/raw.github.com/dpsanders/matplotlib-examples/master/colorline.ipynb
'''

import numpy as np
import matplotlib.pyplot as plt


# Data manipulation:

def make_segments(x, y, z):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


# Interface to LineCollection:

def colorline(x, y, z, c=None,
              cmap=plt.get_cmap('copper'),
              norm=plt.Normalize(),
              linewidth=3,
              linestyle='-',
              alpha=1.0):
    '''
    Plot a colored line with coordinates x and y and z
    Optionally specify colors in the array c
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if c is None:
        c = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        c = np.array([c])

    c = np.asarray(c)

    segments = make_segments(x, y, z)
    lc = Line3DCollection(segments, array=c,
                          cmap=cmap,
                          norm=norm,
                          linewidth=linewidth,
                          linestyle=linestyle,
                          alpha=alpha)

    ax = plt.gca()
    ax.add_collection3d(lc)

    return lc