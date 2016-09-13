#!/bin/bash
# Run the tests for all environment files in this folders. Environment files are
# expected to be like "environment_${key}.yml", where ${key} is the unique key for the
# environment. The conda environment name is expected to be psyplot_${key} with the
# key of the corresponding file.
# The results can be seen in test_psyplot_${key}.log
set -x
envs_dir=`conda info --root`/envs
WORK=`pwd`
for f; do
#     cd ${WORK}
    conda env create -f ${f}
    env=${f%.yml}
    env=psyplot_${env##environment_}
    python_bin=${envs_dir}/${env}/bin/python
    pytest_bin="${envs_dir}/${env}/bin/py.test"
    pip_bin=${envs_dir}/${env}/bin/pip
    conda_bin=${envs_dir}/${env}/bin/conda
    # try to import numpy and if it doesn't work (due to cartopy), update to current version
    ${python_bin} -c "import numpy" > /dev/null
    if [[ $? == 1 ]]; then ${pip_bin} install --upgrade numpy; fi
    # check if psyplot already is installed
    ${python_bin} -c "import psyplot" > /dev/null
    if [[ $? == 1 ]]; then
        source activate ${env}
        conda install --no-deps -c chilipp psyplot
        source deactivate
    fi
    # run tests
    mkdir ${env}
    cd ${env}
    touch matplotlibrc
    ${pytest_bin} --html=../${env}.html --cov=psyplot --cov-report html ../../ &> ../test_${env}.log && mv htmlcov $WORK/cov_${env} && cd $WORK && rm -r ${env} && conda env remove -y -n ${env} || echo "Error occured when testing for ${env}!"
done
