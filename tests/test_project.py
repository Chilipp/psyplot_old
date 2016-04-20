"""Test module of the :mod:`psyplot.project` module"""
import os
import _base_testing as bt
import psyplot.project as psy
import matplotlib.pyplot as plt


class ProjectTester(bt.PsyPlotTestCase):

    plot_type = 'project'

    def test_save_and_load(self):
        """Test project reproducability through the save and load method"""
        plt.close('all')
        maps = psy.plot.mapplot('test-t2m-u-v.nc', name='t2m', time=[1, 2],
                                ax=(2, 2))
        maps[0].update(cmap='winter', bounds='minmax')
        maps.share(keys='bounds')
        grid_ax = plt.subplot2grid((2, 2), (1, 0), 1, 2)
        lines = psy.plot.lineplot('icon_test.nc', name='u', x=0, time=range(5),
                                  ax=grid_ax)
        plt.savefig(os.path.join(bt.ref_dir, self.get_ref_file('save_load1')))
        plt.figure()
        ax = plt.axes(maps[0].plotter.ax.get_position())
        psy.plot.lineplot('icon_test.nc', name='t2m', z=0, x=0, ax=[ax])
        fname = os.path.join(bt.ref_dir, self.get_ref_file('save_load2'))
        plt.savefig(fname)
        p = psy.gcp(True)
        p.save_project(fname + '.pkl')
        p.close(True, True)
        plt.close('all')
        p = psy.Project.load_project(fname + '.pkl')
        plt.figure(1)
        self.compare_figures(self.get_ref_file('save_load1'))
        plt.figure(2)
        self.compare_figures(self.get_ref_file('save_load2'))
        p.close(True, True)
        plt.close('all')

        maps = psy.plot.mapplot('icon_test.nc', name='t2m', time=[1, 2],
                                ax=(2, 2))
        maps[0].update(cmap='winter', bounds='minmax')
        maps.share(keys='bounds')
        grid_ax = plt.subplot2grid((2, 2), (1, 0), 1, 2)
        lines = psy.plot.lineplot('icon_test.nc', name='u', x=0, time=range(5),
                                  ax=grid_ax)

        plt.savefig(os.path.join(bt.ref_dir, self.get_ref_file('save_load3')))
        p.close(True, True)
        plt.close('all')

        p = psy.Project.load_project(fname + '.pkl', alternative_paths={
            '../../../test-t2m-u-v.nc': 'icon_test.nc'})
        plt.figure(1)
        self.compare_figures(self.get_ref_file('save_load3'))
        plt.figure(2)
        self.compare_figures(self.get_ref_file('save_load2'))
        p.close(True, True)
        plt.close('all')
        os.remove(fname + '.pkl')
        for i in range(1, 4):
            os.remove(os.path.join(
                bt.ref_dir, self.get_ref_file('save_load%i' % i)))


if __name__ == '__main__':
    bt.RefTestProgram()
