from psyplot.config.rcsetup import RcParams

rcParams = RcParams(defaultParams={'test': [1, lambda i: int(i)]})
rcParams.update_from_defaultParams()
