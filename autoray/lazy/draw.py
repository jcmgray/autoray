"""Visualizations for ``LazyArray`` computational graphs.
"""

import itertools
import functools


COLORING_SEED = 1  # 8, 10


def set_coloring_seed(seed):
    """Set the seed for the random color generator.

    Parameters
    ----------
    seed : int
        The seed to use.
    """
    global COLORING_SEED
    COLORING_SEED = seed


def hash_to_nvalues(s, nval, seed=None):
    import hashlib

    if seed is None:
        seed = COLORING_SEED

    m = hashlib.sha256()
    m.update(f"{seed}".encode())
    m.update(s.encode())
    hsh = m.hexdigest()

    b = len(hsh) // nval
    if b == 0:
        raise ValueError(
            f"Can't extract {nval} values from hash of length {len(hsh)}"
        )
    return tuple(
        int(hsh[i * b : (i + 1) * b], 16) / 16**b for i in range(nval)
    )


def hash_to_color(
    s,
    hmin=0.0,
    hmax=1.0,
    smin=0.3,
    smax=0.8,
    vmin=0.8,
    vmax=1.0,
):
    """Generate a random color for a string  ``s``.

    Parameters
    ----------
    s : str
        The string to generate a color for.
    hmin : float, optional
        The minimum hue value.
    hmax : float, optional
        The maximum hue value.
    smin : float, optional
        The minimum saturation value.
    smax : float, optional
        The maximum saturation value.
    vmin : float, optional
        The minimum value value.
    vmax : float, optional
        The maximum value value.

    Returns
    -------
    color : tuple
        A tuple of floats in the range [0, 1] representing the RGB color.
    """
    from matplotlib.colors import hsv_to_rgb

    h, s, v = hash_to_nvalues(s, 3)
    h = hmin + h * (hmax - hmin)
    s = smin + s * (smax - smin)
    v = vmin + v * (vmax - vmin)

    return hsv_to_rgb((h, s, v))


def rotated_house_shape(xy, r=0.4):
    x, y = xy
    return [
        [x - r, y - r],
        [x - r, y + r],
        [x, y + r],
        [x + r, y],
        [x, y - r],
    ]


def count_around(c, layout):
    if layout == "wide":
        # just count upwards
        yield from itertools.count(c)
    elif layout == "compact":
        # count backwards, then forwards after reaching zero
        yield from range(c, -1, -1)
        yield from itertools.count(c + 1)
    else:  # 'balanced'
        # count backwards, then forwards, alternating
        step = 0
        # start by stepping to side closer to zero
        sgn = (-1) ** (c <= 0)
        while True:
            cm = c - sgn * step
            if step != 0:  # and (cm >= 0):
                yield cm
            yield c + sgn * step
            step += 1


def get_default_colors_dict(colors):
    import numpy as np

    colors = dict() if colors is None else dict(colors)
    colors.setdefault("None", np.array([0.5, 0.5, 0.5]))
    colors.setdefault("getitem", np.array([0.5, 0.5, 0.5]))
    return colors


def plot_graph(
    self,
    variables=None,
    initial_layout="kamada_kawai",
    dag_spread=2,
    iterations=0,
    k=None,
    color_by="function",
    colors=None,
    connectionstyle="arc3,rad=-0.05",
    arrowsize=6,
    edge_color=(0.5, 0.5, 0.5),
    edge_alpha=0.3,
    var_color=(0, 0.5, 0.25),
    const_color=(0, 0.5, 1.0),
    root_color=(1, 0, 0.5),
    node_shape="s",
    node_scale=1.0,
    node_alpha=1.0,
    show_labels=True,
    label_color=(0.5, 0.5, 0.5),
    label_alpha=1.0,
    font_size=8,
    label_rotation=45,
    figsize=None,
    ax=None,
    show_and_close=True,
    **layout_opts,
):
    """Plot the computational graph of this ``LazyArray``."""
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    if color_by not in ("id", "function", "variables"):
        raise ValueError("color_by must be 'id', 'function' or 'variables'")

    colors = get_default_colors_dict(colors)
    G = self.to_nx_digraph(variables=variables)

    created_fig = ax is None
    if created_fig:
        if figsize is None:
            w = h = (G.number_of_nodes() + 1) ** 0.5
            figsize = (w, h)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        fig.set_facecolor((0, 0, 0, 0))
        ax.axis("off")
        ax.set_aspect("equal")

    node_colors = {}
    node_sizes = {}
    node_labels = {}
    node_markers = {}
    for node in G.nodes:
        # set node color
        if node is self:
            node_markers[node] = "X"

        if color_by == "variables":
            if node is self:
                node_colors[node] = root_color
            elif G.nodes[node]["variable"]:
                node_colors[node] = var_color
            else:
                node_colors[node] = const_color

        elif color_by == "function":
            if node.fn_name in colors:
                node_colors[node] = colors[node.fn_name]
            else:
                node_colors[node] = hash_to_color(node.fn_name)

        elif color_by == "id":
            node_colors[node] = hash_to_color(str(id(node)))

        # set node size
        node_sizes[node] = 6 * node_scale * (np.log2(node.size) + 1)

        # set node label and marker
        if node.fn_name == "None":
            node_markers.setdefault(node, "o")
            node_labels[node] = ""
        if node.fn_name == "getitem":
            node_markers.setdefault(node, ".")
            node_labels[node] = ""
        else:
            node_labels[node] = node.fn_name

        node_markers.setdefault(node, node_shape)

    # compute a layout for the graph
    if initial_layout == "layers":
        for layer, nodes in enumerate(nx.topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer

        layout_opts.setdefault("align", "vertical")
        pos = nx.multipartite_layout(G, subset_key="layer", **layout_opts)

        if layout_opts["align"] == "horizontal":
            dag_spread = 1 / dag_spread
        for k, (x, y) in pos.items():
            pos[k] = (x, dag_spread * y)

    else:
        if initial_layout == "spiral":
            layout_opts.setdefault("equidistant", True)

        pos = getattr(nx, initial_layout + "_layout")(G, **layout_opts)

    # further spring based refinement
    if iterations:
        pos = nx.layout.spring_layout(G, pos=pos, k=k, iterations=iterations)

    # draw edges!
    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        edge_color=edge_color,
        alpha=edge_alpha,
        connectionstyle=connectionstyle,
        arrowsize=arrowsize,
        arrows=True,
    )
    # draw nodes!
    for node in G.nodes:
        ax.scatter(
            *pos[node],
            s=node_sizes[node],
            facecolor=node_colors[node],
            alpha=node_alpha,
            marker=node_markers[node],
        )
    if show_labels:
        # draw labels!
        text = nx.draw_networkx_labels(
            G,
            pos=pos,
            ax=ax,
            labels=node_labels,
            font_color=label_color,
            font_size=font_size,
            alpha=label_alpha,
            bbox={"color": (0, 0, 0, 0)},
        )
        for _, t in text.items():
            t.set_rotation(label_rotation)

    if (fig is not None) and show_and_close:
        plt.show()
        plt.close(fig)

    return fig, ax


def plot_circuit(
    self,
    color_by="function",
    colors=None,
    layout="balanced",
    linewidth=None,
    linewidth_scale=1,
    linealpha=1.0,
    fontsize=None,
    fontsize_scale=1,
    figsize=None,
    ax=None,
    show_and_close=True,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if color_by not in ("id", "function"):
        raise ValueError("color_by must be 'id' or 'function'")
    if layout not in ("balanced", "compact", "wide"):
        raise ValueError("layout must be 'balanced', 'compact', or 'wide'")

    colors = get_default_colors_dict(colors)

    nodes = list(self.ascend())
    steps = {node: i for i, node in enumerate(nodes)}
    rails = {self: 0}
    edges = []
    active = {0}

    for node in reversed(nodes):
        if color_by == "function":
            if node.fn_name in colors:
                c = colors[node.fn_name]
            else:
                c = hash_to_color(node.fn_name)
        else:
            c = hash_to_color(str(id(node)))
        colors[node] = c

        # free up the column
        active.remove(rails[node])

        # want to plot in same order the computational graph was created
        deps = sorted(node.deps, key=lambda x: -x.depth)

        # get the 'nearest columns' that are available for children
        close_rails = (
            c for c in count_around(rails[node], layout) if c not in active
        )
        child_rails = (next(close_rails) for c in deps if c not in rails)

        for child in deps:
            if child not in rails:
                # place the node
                rails[child] = next(child_rails)
                active.add(rails[child])
            # add connector
            edges.append((node, child))

    created_fig = ax is None
    if created_fig:
        if figsize is None:
            w = h = (len(nodes) + 1) ** (2 / 3)
            figsize = (w, h)
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor((0, 0, 0, 0))
        ax.axis("off")
        ax.set_aspect("equal")

    if linewidth is None:
        linewidth = linewidth_scale * 8 * (figsize[1] / len(nodes))
    if fontsize is None:
        fontsize = fontsize_scale * 40 * (figsize[1] / len(nodes))

    # draw the edges
    for a, b in edges:
        xya = steps[a], rails[a]
        xyb = steps[b], rails[b]

        if b.fn_name == "getitem":
            color = colors[b.deps[0]]
        else:
            color = colors[b]

        path_opts = dict(
            edgecolor=color,
            linewidth=linewidth,
            alpha=linealpha,
            facecolor="none",
            zorder=9,
        )

        if xya[1] == xyb[1]:
            # straight line
            xy = (xya[0], xyb[0])
            patch = mpl.patches.PathPatch(
                mpl.path.Path(
                    [xya, xyb], [mpl.path.Path.MOVETO, mpl.path.Path.LINETO]
                ),
                **path_opts,
            )
        else:
            # right angle line
            patch = mpl.patches.PathPatch(
                mpl.path.Path(
                    [
                        xya,
                        (xya[0], xyb[1] + 0.25 * (-1) ** (xya[1] < xyb[1])),
                        (xya[0] - 0.25, xyb[1]),
                        xyb,
                    ],
                    [mpl.path.Path.MOVETO] + [mpl.path.Path.LINETO] * 3,
                ),
                **path_opts,
            )
        ax.add_patch(patch)

    # draw the nodes, and figure out plot range
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for node in nodes:
        xy = steps[node], rails[node]
        xmin, xmax = min(xmin, xy[0]), max(xmax, xy[0])
        ymin, ymax = min(ymin, xy[1]), max(ymax, xy[1])
        if not node.deps:
            # make a square patch centered at xy with radius 0.4
            patch = mpl.patches.Circle(
                xy=xy, radius=0.4, color=colors[node], zorder=10
            )
        elif node.fn_name == "getitem":
            # make a small circle for getitem (since not really a node)
            patch = mpl.patches.Circle(
                xy=xy, radius=0.15, color=colors[node.deps[0]], zorder=10
            )
        else:
            # make a 'rotated house' shape
            patch = mpl.patches.Polygon(
                rotated_house_shape(xy, r=0.3), color=colors[node], zorder=10
            )
        ax.add_patch(patch)

    # draw the labels
    for node in nodes:
        name = "←" if node.fn_name == "None" else node.fn_name
        color = colors[node]
        ax.text(
            steps[node] - 0.25,
            ymax + 1.0,
            f"{name}{list(node.shape)}",
            ha="left",
            va="bottom",
            color=color,
            fontsize=fontsize,
            rotation=45,
        )
        ax.plot(
            [steps[node], steps[node]],
            [ymax + 1, rails[node]],
            color=color,
            linewidth=linewidth / 2,
            alpha=0.25,
            linestyle=":",
            clip_on=False,
        )

    # set plot limits
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)

    if (fig is not None) and show_and_close:
        plt.show()
        plt.close(fig)

    return fig, ax


# a style to use for matplotlib that works with light and dark backgrounds
NEUTRAL_STYLE = {
    "axes.edgecolor": (0.5, 0.5, 0.5),
    "axes.facecolor": (0, 0, 0, 0),
    "axes.grid": True,
    "axes.labelcolor": (0.5, 0.5, 0.5),
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.facecolor": (0, 0, 0, 0),
    "grid.alpha": 0.1,
    "grid.color": (0.5, 0.5, 0.5),
    "legend.frameon": False,
    "text.color": (0.5, 0.5, 0.5),
    "xtick.color": (0.5, 0.5, 0.5),
    "xtick.minor.visible": True,
    "ytick.color": (0.5, 0.5, 0.5),
    "ytick.minor.visible": True,
}


def default_to_neutral_style(fn):
    """Wrap a function or method to use the neutral style by default."""

    @functools.wraps(fn)
    def wrapper(*args, style="neutral", **kwargs):
        import matplotlib.pyplot as plt

        if style == "neutral":
            style = NEUTRAL_STYLE
        elif not style:
            style = {}

        with plt.style.context(style):
            return fn(*args, **kwargs)

    return wrapper


@default_to_neutral_style
def plot_history_size_footprint(
    self,
    log=None,
    figsize=(8, 2),
    color="purple",
    alpha=0.5,
    rasterize=4096,
    rasterize_dpi=300,
    ax=None,
    show_and_close=True,
):
    """Plot the memory footprint throughout this computation.

    Parameters
    ----------
    log : None or int, optional
        If not None, display the sizes in base ``log``.
    figsize : tuple, optional
        Size of the figure.
    color : str, optional
        Color of the line.
    alpha : float, optional
        Alpha of the line.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, will be created if not provided.
    return_fig : bool, optional
        If True, return the figure object, else just show and close it.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    y = np.array(self.history_size_footprint())
    if log:
        y = np.log2(y) / np.log2(log)
        ylabel = f"$\\log_{log}[total size]$"
    else:
        ylabel = "total size"

    x = np.arange(y.size)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_dpi(rasterize_dpi)
    else:
        fig = None

    if isinstance(rasterize, (float, int)):
        # only turn on above a certain size
        rasterize = y.size > rasterize

    if rasterize:
        ax.set_rasterization_zorder(0)

    ax.fill_between(x, 0, y, alpha=alpha, color=color, zorder=-1)

    if fig is not None:
        ax.grid(True, c=(0.95, 0.95, 0.95), which="both")
        ax.set_axisbelow(True)
        ax.set_xlim(0, np.max(x))
        ax.set_ylim(0, np.max(y))
        ax.set_ylabel(ylabel)

    if (fig is not None) and show_and_close:
        plt.show()
        plt.close(fig)

    return fig, ax


@default_to_neutral_style
def plot_history_functions(
    self,
    *,
    fn=None,
    log=None,
    colors=None,
    kind="lines",
    scatter_size=5,
    scatter_marker="s",
    lines_width=5,
    image_alpha_pow=2 / 3,
    image_aspect=4,
    legend=True,
    legend_ncol=None,
    legend_bbox_to_anchor=None,
    legend_loc=None,
    rasterize=4096,
    rasterize_dpi=300,
    ax=None,
    figsize=(8, 2),
    show_and_close=True,
):
    """Plot the functions used throughout this computation, color coded, as
    either a scatter plot or an image, showing the size of the that individual
    intermediate as well.
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if fn is not None:
        ylabel = "custom"
    else:
        ylabel = "node size"

        def fn(node):
            return node.size

    if log:
        # wrap the function to take log of values
        ylabel = f"$\\log_{{{log}}}[{ylabel}]$"
        orig_fn = fn

        def fn(node):
            return np.log2(orig_fn(node)) / np.log2(log)

    colors = get_default_colors_dict(colors)

    xs = []
    ys = []
    cs = []
    ymax = 0
    for i, node in enumerate(self.ascend()):
        xs.append(i)
        y = fn(node)
        ymax = max(ymax, y)
        ys.append(y)
        try:
            c = colors[node.fn_name]
        except KeyError:
            c = colors[node.fn_name] = hash_to_color(node.fn_name)
        cs.append(c)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_dpi(rasterize_dpi)
        ax.set_ylabel(ylabel)
    else:
        fig = None

    if isinstance(rasterize, (float, int)):
        # only turn on above a certain size
        rasterize = len(xs) > rasterize

    if rasterize:
        ax.set_rasterization_zorder(0)

    if kind == "scatter":
        ax.scatter(
            xs,
            ys,
            c=cs,
            s=scatter_size,
            marker=scatter_marker,
            rasterized=rasterize,
        )

    elif kind == "lines":
        lns = [((x, 0.0), (x, y)) for x, y in zip(xs, ys)]
        ax.add_collection(
            mpl.collections.LineCollection(
                lns,
                colors=cs,
                zorder=-1,
                lw=lines_width,
            )
        )
        ax.set_xlim(-0.5, len(lns) + 0.5)
        ax.set_ylim(0, 1.05 * ymax)

    elif kind == "image":
        ax.axis("off")
        ys = np.array(ys)
        ys = (ys / ys.max()).reshape(-1, 1) ** image_alpha_pow
        N = len(cs)
        da = round((N / image_aspect) ** 0.5)
        db = N // da
        while da * db < N:
            db += 1
        Ns = da * db
        img = np.concatenate([cs, ys], axis=1)
        img = np.concatenate([img, np.tile(0.0, (Ns - N, 4))], axis=0)
        img = img.reshape(da, db, 4)
        ax.imshow(img, zorder=-1)

    if legend:
        legend_items = [
            mpl.patches.Patch(facecolor=c, label=fn_name)
            for fn_name, c in colors.items()
        ]

        if legend_ncol is None:
            legend_ncol = max(1, round(len(legend_items) / 6))
        if legend_bbox_to_anchor is None:
            legend_bbox_to_anchor = (1.0, 1.0)
        if legend_loc is None:
            legend_loc = "upper left"

        ax.legend(
            handles=legend_items,
            ncol=legend_ncol,
            bbox_to_anchor=legend_bbox_to_anchor,
            loc=legend_loc,
        )

    if (fig is not None) and show_and_close:
        plt.show()
        plt.close(fig)

    return fig, ax


@default_to_neutral_style
def plot_history_stats(
    self,
    *,
    fn="count",
    colors=None,
    rasterize_dpi=300,
    ax=None,
    figsize=(2, 2),
    show_and_close=True,
):
    from matplotlib import pyplot as plt

    stats = self.history_stats(fn)

    colors = get_default_colors_dict(colors)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_dpi(rasterize_dpi)
    else:
        fig = None

    xs, labels, clrs = [], [], []

    for fn_name, cnt in sorted(stats.items(), key=lambda x: -x[1]):
        xs.append(cnt)
        labels.append(f"{fn_name}: {cnt}")
        try:
            color = colors[fn_name]
        except KeyError:
            color = colors[fn_name] = hash_to_color(fn_name)
        clrs.append(color)

    ax.pie(x=xs, labels=labels, colors=clrs)

    if (fig is not None) and show_and_close:
        plt.show()
        plt.close(fig)

    return fig, ax
