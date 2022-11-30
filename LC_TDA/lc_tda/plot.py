import matplotlib.pyplot as plt
import matplotlib


def rcparams(r=1.0, use_tex=False):
    """A function to format matplotlib plots.

    Args:
        r (float, optional): ratio of the plot. Defaults to 1.0.
        use_tex (bool, optional): latex symbols. Defaults to False.
    """
    if use_tex:
        matplotlib.rcParams['text.usetex'] = True
    else:
        matplotlib.rcParams['text.usetex'] = False

    # font
    matplotlib.rcParams['font.size'] = 12.0 * r
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # lines
    matplotlib.rcParams['lines.linewidth'] = 2.0

    # axes setting
    matplotlib.rcParams['axes.labelsize'] = 20.0 * r
    matplotlib.rcParams['axes.linewidth'] = 1.0
    matplotlib.rcParams['axes.titlesize'] = 20.0 * r
    matplotlib.rcParams['axes.axisbelow'] = True

    # ticks
    matplotlib.rcParams['xtick.major.size'] = 0.0
    matplotlib.rcParams['xtick.minor.size'] = 0.0
    matplotlib.rcParams['xtick.major.pad'] = 6.0
    matplotlib.rcParams['xtick.minor.pad'] = 6.0
    matplotlib.rcParams['xtick.color'] = '333333'
    matplotlib.rcParams['xtick.labelsize'] = 20.0 * r
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.major.size'] = 0.0
    matplotlib.rcParams['ytick.minor.size'] = 0.0
    matplotlib.rcParams['ytick.major.pad'] = 6.0
    matplotlib.rcParams['ytick.minor.pad'] = 6.0
    matplotlib.rcParams['ytick.color'] = '333333'
    matplotlib.rcParams['ytick.labelsize'] = 20.0 * r
    matplotlib.rcParams['ytick.direction'] = 'in'

    # grid
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.alpha'] = 0.3
    matplotlib.rcParams['grid.linewidth'] = 1

    # legend
    matplotlib.rcParams['legend.title_fontsize'] = 15.0 * r
    matplotlib.rcParams['legend.fontsize'] = 15.0 * r
    matplotlib.rcParams['legend.fancybox'] = True
    matplotlib.rcParams['legend.facecolor'] = 'fdfdfd'

    # figure
    matplotlib.rcParams['figure.figsize'] = [8.0*r, 8.0*r]
    matplotlib.rcParams['figure.facecolor'] = '1.0'
    matplotlib.rcParams['figure.edgecolor'] = '0.5'
    matplotlib.rcParams['figure.dpi'] = 75

    # bar plot
    matplotlib.rcParams['hatch.linewidth'] = 0.5

    # animation
    matplotlib.rcParams['animation.writer'] = 'html'
    matplotlib.rcParams['animation.html'] = 'html5'

    # color
    matplotlib.rcParams['axes.prop_cycle'] = plt.cycler(color=['#515151', '#df5048', '#3370d8', '#5baa71',
                                                               '#a87bd8', '#c49b33', '#5bc8ca', '#76504f',
                                                               '#8e8c2b', '#ea6f2d', '#7099c8', '#80b537'])


def format_axis(ax):
    """A function to format matplotlib axis.

    Args:
        ax (matplotlib.pyplot.axis): axis.
    """
    from matplotlib.ticker import (AutoMinorLocator)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)


def format_axis_im(ax):
    """A function to format matplotlib axis for imshow.

    Args:
        ax (matplotlib.pyplot.axis): axis.
    """
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=6)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
