from psyplot.config.rcsetup import RcParams, validate_dict

rcParams = RcParams(defaultParams={
    'test': [1, lambda i: int(i)],
    'project.plotters': [
        {'test_plotter': {
            'module': 'psyplot_test.plotter',
            'plotter_name': 'TestPlotter', 'import_plotter': True}},
        validate_dict]})
rcParams.update_from_defaultParams()
