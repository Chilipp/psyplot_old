About psyplot
=============

Why psyplot
-----------
The main idea for the psyplot package arose from the need of the author to
create plots interactively when developing and analysing climate models. There
is a wide range of software tools available for visualizing raster data, but
they often are difficult to use in a script or have low flexibility or are far
to complex for a simple visualization. Therefore, the choice was to use the
python package matplotlib_. With python we have an interactive and developping
programming language and matplotlib allows a very flexible handling of data
visualization with a very convincing and well documented API.

However, matplotlib can be quite costly in terms of time when modifying the
plot for specific needs. Therefore it's use very often results in long scripts
of about hundreds of lines and the reproduction of one specific change in
another plot can be very tidious. Furthermore, although matplotlib in
principle allows interactive usage, it can be very unconfortable because every
artist has to be stored in a variable in order to be modified (if that works at all).

This finally gave the motiviation to pack together the different scripts into
one framework that can be used for interactive visualization without minimizing
the extensive visualization capabilities of matplotlib. Furthermore the
framework allows to easily manage and format multiple plots and enables the
User to automatically apply the same settings to multiple datasets.

But try it and :ref:`get started <getting-started>`.

.. _matplotlib: http://matplotlib.org


About the author
----------------
`Philipp Sommer`_ works as a PhD student for climate modelling in the
Arve-Regolith-Vegetation (ARVE) group in the Institute of Earth Surface
Dynamics (IDYST) at the University of Lausanne. He has done his master in
Integrated Climate System Science at the University of Hamburg and a Bachelor
in Phyiscs at the University of Heidelberg. His master thesis focused on the
development of an irrigation scheme for the land-surface scheme JSBACH of the
`Max-Planck-Institute Earth-System-Model (MPI-ESM)`_. Having worked for two
years as a student helper in the working group on Terrestrial Hydrology at the
MPI with Stefan Hagemann, he mainly focused on the evaluation of climate model
data of CMIP5 models but also on the work with the `ICON model`_.
The latter basicly was the motivation for the visualization package psyplot.

.. _Philipp Sommer: http://arve.unil.ch/people/philipp-sommer
.. _Max-Planck-Institute Earth-System-Model (MPI-ESM): http://www.mpimet.mpg.de/en/science/models/mpi-esm.html
.. _ICON model: http://www.mpimet.mpg.de/en/science/models/icon.html


License
-------
psyplot is published under the
`GNU General Public License v2.0 <http://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html>`__