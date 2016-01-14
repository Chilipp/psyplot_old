#!/bin/bash
# script to automatically generate the psyplot api documentation using
# sphinx-apidoc and sed
sphinx-apidoc -f -M -e  -T -o api ../psyplot/ ../plotter/common.py
# replace chapter title in psyplot.rst
sed -i '' -e 1,1s/.*/'API Reference'/ api/psyplot.rst
# add imported members at the top level module
sed -i '' -e /Subpackages/'i\'$'\n'".. autosummary:: \\
\    ~psyplot.config.rcsetup.rcParams \\
\    ~psyplot.data.InteractiveArray \\
\    ~psyplot.data.InteractiveList \\
    \\
    " api/psyplot.rst

for f in `ls api/*.rst`; do
    sed -i '' '/\s*:undoc-members:/d' ${f}
done