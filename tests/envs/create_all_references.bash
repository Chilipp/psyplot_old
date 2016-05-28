#!/bin/bash
# Create the references for all environment files in this folders. Environment files are
# expected to be like "environment_${key}.yml", where ${key} is the unique key for the 
# environment. The conda environment name is expected to be psyplot_${key} with the
# key of the corresponding file. 
# The results can be seen in ref_psyplot_${key}.log
envs_dir=`conda info --root`/envs
for f; do

	conda env create -f ${f}
	env=${f%.yml}
	env=psyplot_${env##environment_}
	python_bin=${envs_dir}/${env}/bin/python
	pip_bin=${envs_dir}/${env}/bin/pip
	conda_bin=${envs_dir}/${env}/bin/conda
	# try to import numpy and if it doesn't work (due to cartopy), update to current version
	${python_bin} -c "import numpy" > /dev/null
	if [[ $? == 1 ]]; then ${pip_bin} install --upgrade numpy; fi
	# check if psyplot already is installed
	${python_bin} -c "import psyplot" > /dev/null
	if [[ $? == 1 ]]; then 
	    source activate ${env}
	    conda install -c chilipp psyplot
	    source deactivate
    fi
	# create reference figures
	${python_bin} ../create_references.py &> ref_${env}.log && conda env remove -y -n ${env} || echo "Error occured when creating references for ${env}!" &
	
done
