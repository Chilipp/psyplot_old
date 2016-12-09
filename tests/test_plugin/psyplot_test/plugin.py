from psyplot.config.rcsetup import RcParams, validate_dict

rcParams = RcParams(defaultParams={
    'test': [1, lambda i: int(i)],
    'project.plotters': [{'test_plotter': {}}, validate_dict]})
rcParams.update_from_defaultParams()
