
# coding: utf-8

# Available colormaps
# ===========
# This example shows you how you can explore your colormaps
# 
# It uses the `psyplot.plotter.colors.show_colormaps` function and can visualize all, selected, or your own colormaps.

# In[ ]:

import psyplot.project as psy
# get_ipython().magic('matplotlib inline')


# You can either visualize specific colormaps

# In[ ]:

psy.show_colormaps('RdBu', 'coolwarm', 'viridis')


# display your own ones

# In[ ]:

import matplotlib.colors as mcol
cmap = mcol.LinearSegmentedColormap.from_list(
    'my_cmap', [[1, 0, 0], [0, 1, 0], [0, 0, 1]], N=11)
psy.show_colormaps(cmap)


# or all that are avaiable

# In[ ]:

psy.show_colormaps()


# Those colormaps (or their name) can then be used for the `cmap` formatoption or the `color` formatoption.
