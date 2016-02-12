import matplotlib as mpl


mpl_version = float('.'.join(mpl.__version__.split('.')[:2]))

if mpl_version >= 1.5:
    from matplotlib.font_manager import weight_dict
    bold = weight_dict['bold']
else:
    bold = 'bold'
