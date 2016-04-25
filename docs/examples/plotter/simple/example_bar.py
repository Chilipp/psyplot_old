
# coding: utf-8

# Bar plot demo
# =========
# This example shows you how to make a bar plot using the `psyplot.project.ProjectPlotter.barplot` method.

# In[ ]:

import psyplot.project as psy
# get_ipython().magic(u'matplotlib inline')
# get_ipython().magic(u'config InlineBackend.close_figures = False')


# In[ ]:

axes = iter(psy.multiple_subplots(2, 2, n=3))
for var in ['t2m', 'u', 'v']:
    psy.plot.barplot(
        'demo.nc',  # netCDF file storing the data
        name=var, # one plot for each variable
        y=[0, 1],  # two bars in total
        z=0, x=0,      # choose latitude and longitude as dimensions
        ylabel="{desc}",  # use the longname and units on the y-axis
        ax=next(axes),
        color='coolwarm', legend=False, xticklabels='%B %Y'
    )
bars = psy.gcp(True)
bars.show()


# In[ ]:

bars.close(True, True)

