#!/bin/bash
# Create the references for all environment files in this folders. Environment files are
# expected to be like "environment_${key}.yml", where ${key} is the unique key for the 
# environment. The conda environment name is expected to be psyplot_${key} with the
# key of the corresponding file. 
# The results can be seen in ref_psyplot_${key}.log
for f in `ls environment_*.yml`; do

	conda env create -f ${f}
	env=${f%.yml}
	env=psyplot_${env##environment_}
	source activate ${env}
	# check if psyplot already is installed
	python -c "import psyplot" > /dev/null
	if [[ $? == 1 ]]; then conda install -c chilipp psyplot; fi
	# create reference figures
	python ../create_references.py &> ref_${env}.log || echo "Error occured when creating references for ${env}!" && source deactivate && conda env remove -y -n ${env} &
	
done