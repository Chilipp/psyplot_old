
# coding: utf-8

# Usage of Climate Data Operators
# =================
# This example shows you how CDOs are binded in the psyplot package

# In[ ]:


import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)


# In[ ]:

import psyplot.project as psy
# get_ipython().magic('matplotlib inline')


# In[ ]:

cdo = psy.Cdo()
lines = cdo.fldmean(input='-sellevidx,1 demo.nc', returnLine='t2m')
lines.update(xticks='month', xticklabels='%b %Y')


# In[ ]:

maps = cdo.timmean(input='demo.nc', returnMap='t2m')
maps.update(cmap='RdBu_r')


# In[ ]:

psy.gcp(True).close(True, True)

