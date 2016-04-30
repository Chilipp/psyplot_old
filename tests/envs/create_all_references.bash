#!/bin/bash
# Create the references for all environment files in this folders. Environment files are
# expected to be like "environment_${key}.yml", where ${key} is the unique key for the 
# environment. The conda environment name is expected to be psyplot_${key} with the
# key of the corresponding file. 
# The results can be seen in ref_psyplot_${key}.log
envs_dir=`conda info --root`/envs
for f in `ls environment_*.yml`; do

	conda env create -f ${f}
	env=${f%.yml}
	env=psyplot_${env##environment_}
	python_bin=${envs_dir}/${env}/bin/python
	pip_bin=${envs_dir}/${env}/bin/pip
	conda_bin=${envs_dir}/${env}/bin/conda
	# check if psyplot already is installed
	${python_bin} -c "import psyplot" > /dev/null
	if [[ $? == 1 ]]; then ${conda_bin} install -c chilipp psyplot; fi
	# update numpy manually because this is sometimes lacking due to cartopy
	update_numpy=`python -c "import numpy, re; print(list(map(int, re.findall('\d+', numpy.__version__)))[:2] < [1, 11])"`
	if [[ $update_numpy == "True" ]]; then ${pip_bin} install --upgrade numpy; fi
	# create reference figures
	${python_bin} ../create_references.py &> ref_${env}.log && conda env remove -y -n ${env} || echo "Error occured when creating references for ${env}!" &
	
done