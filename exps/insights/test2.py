from functools import partial
import itertools

from cycler import cycler

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as mticker

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from functools import partial
import itertools
from sklearn.preprocessing import MinMaxScaler

def filled_hist(ax, edges, values, bottoms=None, orientation='v',
                **kwargs):
    """
    Draw a histogram as a stepped patch.

    Parameters
    ----------
    ax : Axes
        The axes to plot to

    edges : array
        A length n+1 array giving the left edges of each bin and the
        right edge of the last bin.

    values : array
        A length n array of bin counts or values

    bottoms : float or array, optional
        A length n array of the bottom of the bars.  If None, zero is used.

    orientation : {'v', 'h'}
       Orientation of the histogram.  'v' (default) has
       the bars increasing in the positive y-direction.

    **kwargs
        Extra keyword arguments are passed through to `.fill_between`.

    Returns
    -------
    ret : PolyCollection
        Artist added to the Axes
    """
    print(orientation)
    if orientation not in 'hv':
        raise ValueError(f"orientation must be in {{'h', 'v'}} "
                         f"not {orientation}")

    kwargs.setdefault('step', 'post')
    kwargs.setdefault('alpha', 0.7)
    edges = np.asarray(edges)
    values = np.asarray(values)
    if len(edges) - 1 != len(values):
        raise ValueError(f'Must provide one more bin edge than value not: '
                         f'{len(edges)=} {len(values)=}')

    if bottoms is None:
        bottoms = 0
    bottoms = np.broadcast_to(bottoms, values.shape)

    values = np.append(values, values[-1])
    bottoms = np.append(bottoms, bottoms[-1])
    if orientation == 'h':
        return ax.fill_betweenx(edges, values, bottoms,
                                **kwargs)
    elif orientation == 'v':
        return ax.fill_between(edges, values, bottoms,
                               **kwargs)
    else:
        raise AssertionError("you should never be here")


def stack_hist(ax, stacked_data, sty_cycle, bottoms=None,
               hist_func=None, labels=None,
               plot_func=None, plot_kwargs=None):
    """
    Parameters
    ----------
    ax : axes.Axes
        The axes to add artists too

    stacked_data : array or Mapping
        A (M, N) shaped array.  The first dimension will be iterated over to
        compute histograms row-wise

    sty_cycle : Cycler or operable of dict
        Style to apply to each set

    bottoms : array, default: 0
        The initial positions of the bottoms.

    hist_func : callable, optional
        Must have signature `bin_vals, bin_edges = f(data)`.
        `bin_edges` expected to be one longer than `bin_vals`

    labels : list of str, optional
        The label for each set.

        If not given and stacked data is an array defaults to 'default set {n}'

        If *stacked_data* is a mapping, and *labels* is None, default to the
        keys.

        If *stacked_data* is a mapping and *labels* is given then only the
        columns listed will be plotted.

    plot_func : callable, optional
        Function to call to draw the histogram must have signature:

          ret = plot_func(ax, edges, top, bottoms=bottoms,
                          label=label, **kwargs)

    plot_kwargs : dict, optional
        Any extra keyword arguments to pass through to the plotting function.
        This will be the same for all calls to the plotting function and will
        override the values in *sty_cycle*.

    Returns
    -------
    arts : dict
        Dictionary of artists keyed on their labels
    """
    # deal with default binning function
    if hist_func is None:
        hist_func = np.histogram

    # deal with default plotting function
    if plot_func is None:
        plot_func = filled_hist

    # deal with default
    if plot_kwargs is None:
        plot_kwargs = {}
    print(plot_kwargs)
    try:
        l_keys = stacked_data.keys()
        label_data = True
        if labels is None:
            labels = l_keys

    except AttributeError:
        label_data = False
        if labels is None:
            labels = itertools.repeat(None)

    if label_data:
        loop_iter = enumerate((stacked_data[lab], lab, s)
                              for lab, s in zip(labels, sty_cycle))
    else:
        loop_iter = enumerate(zip(stacked_data, labels, sty_cycle))

    arts = {}
    for j, (data, label, sty) in loop_iter:
        if label is None:
            label = f'dflt set {j}'
        label = sty.pop('label', label)
        vals, edges = hist_func(data)
        if bottoms is None:
            bottoms = np.zeros_like(vals)
        top = bottoms + vals
        print(sty)
        sty.update(plot_kwargs)
        print(sty)
        ret = plot_func(ax, edges, top, bottoms=bottoms,
                        label=label, **sty)
        bottoms = top
        arts[label] = ret
    ax.legend(fontsize=10)
    return arts


# # set up histogram function to fixed bins
# edges = np.linspace(-3, 3, 20, endpoint=True)
# hist_func = partial(np.histogram, bins=edges)

# # set up style cycles
# color_cycle = cycler(facecolor=plt.rcParams['axes.prop_cycle'][:4])
# label_cycle = cycler(label=[f'set {n}' for n in range(4)])
# hatch_cycle = cycler(hatch=['/', '*', '+', '|'])

# # Fixing random state for reproducibility
# np.random.seed(19680801)

# stack_data = np.random.randn(4, 12250)
# dict_data = dict(zip((c['label'] for c in label_cycle), stack_data))



zc_name_list = ['plain_layerwise', 'snip_layerwise',
                'grad_norm_layerwise', 'fisher_layerwise', 'l2_norm_layerwise', 'grasp_layerwise']


custom_colors = ['#e377c2', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Your original code's data processing section would change to create stack_data
stack_data = []
for i, zc_name in enumerate(zc_name_list):
    data = np.loadtxt(f'gbdt_{zc_name}.csv', delimiter=',', skiprows=1)
    feature_importance = data[:, 1]
    # Normalize feature importance
    scaler = MinMaxScaler()
    feature_importance_normalized = scaler.fit_transform(feature_importance.reshape(-1, 1)).flatten()
    stack_data.append(feature_importance_normalized)

# Convert stack_data to a NumPy array for processing with the stack_hist function
stack_data = np.array(stack_data)

# Prepare the edges of the bins based on the layer indexes (assuming these are your bins)
edges = np.arange(stack_data.shape[1] + 1)

# Use the partial function to fix the edges
hist_func = partial(np.histogram, bins=edges)

# Use the stack_hist function to plot
fig, ax = plt.subplots()
color_cycle = cycler(facecolor=custom_colors)
hatch_cycle = cycler(hatch=[None] * len(zc_name_list))  # No hatch patterns
sty_cycle = (color_cycle + hatch_cycle)

# Assuming that you have labels in zc_name_list
arts = stack_hist(ax, stack_data.T, sty_cycle=sty_cycle,
                  hist_func=hist_func, labels=zc_name_list)

ax.set_yscale('log')
ax.set_xlabel('Layer Index', fontsize=18)
ax.set_ylabel('Relative Importance (Normalized)', fontsize=18)
ax.set_xticks(np.arange(min(edges), max(edges), 20))
ax.set_xticklabels(np.arange(min(edges), max(edges), 20), fontsize=14, rotation=45)
plt.tight_layout()
plt.savefig('./gbdt_streamgraph_combined.png')