from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt

# import textwrap
import numpy as np
import scanpy as sc

# import seaborn as sns
from matplotlib import rcParams
from matplotlib.axes import Axes

# import matplotlib.pyplot as plt
# import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

# import pandas as pd
from matplotlib.patches import Patch

# from upsetplot import plot, from_contents
# from itertools import chain
from scanpy._utils import Empty, _empty
from scanpy.pl._tools.scatterplots import (
    _check_crop_coord,
    _check_img,
    _check_na_color,
    _check_scale_factor,
    _check_spatial_data,
    _check_spot_size,
)
from scanpy.pl._utils import (
    ColorLike,
)


def heatmap_topic(adata, groupby=None, topics=None, figsize=(10, 5), cmap=None):
    topic_prop = adata.obs.copy()

    if topics is None:
        topics = topic_prop.columns[topic_prop.columns.str.startswith("Topic")]
        topic_prop = topic_prop.loc[:, topics]
    else:
        topic_prop = topic_prop.loc[:, topics]

    topic_adata = ad.AnnData(topic_prop)
    sc.pl.clustermap(topic_adata, obs_key=groupby, figsize=figsize, cmap=cmap)


def heatmap(
    adata,
    topic_prop,
    groupby,
    dendrogram=False,
    swap_axes=True,
    cmap=None,
    figsize=(10, 5),
    save=None,
    **kwargs,
):
    """Creates a heatmap of cell(grouped by groupby) by topics

    Args:
        adata (_type_): Adata object
        topic_prop (_type_): Topic proportions returned by STAMP
        groupby (_type_, optional): Column in adata object to arrange the cells
        figsize (tuple, optional): Figsize. Defaults to (10, 5).
        dendrogram (bool, optional): Whether to cluster. Defaults to False.
        swap_axes (bool, optional): Whether to swap x and y axis. Defaults to True.
        cmap (_type_, optional): What matplotib cmap to use. Defaults to None.
        save (_type_, optional): Whether to save the object. Defaults to None.
    """
    topic_adata = ad.AnnData(topic_prop)
    topic_adata.obs[groupby] = adata.obs[groupby]

    fig = sc.pl.heatmap(
        adata=topic_adata,
        var_names=topic_prop.columns,
        groupby=groupby,
        figsize=figsize,
        dendrogram=dendrogram,
        swap_axes=swap_axes,
        cmap=cmap,
        save=False,
        # show=False,
        **kwargs,
    )
    return fig


def matrixplot(
    adata,
    topic_prop,
    groupby,
    dendrogram=False,
    swap_axes=True,
    cmap=None,
    figsize=(10, 5),
    save=None,
    **kwargs,
):
    """Creates a matrixplot of cell(grouped by groupby) by topics

    Args:
        adata (_type_): Adata object
        topic_prop (_type_): Topic proportions returned by STAMP
        groupby (_type_, optional): Column in adata object to arrange the cells
        figsize (tuple, optional): Figsize. Defaults to (10, 5).
        dendrogram (bool, optional): Whether to cluster. Defaults to False.
        swap_axes (bool, optional): Whether to swap x and y axis. Defaults to True.
        cmap (_type_, optional): What matplotib cmap to use. Defaults to None.
        save (_type_, optional): Whether to save the object. Defaults to None.
    """

    topic_adata = ad.AnnData(topic_prop)
    topic_adata.obs[groupby] = adata.obs[groupby]
    sc.pl.matrixplot(
        adata=topic_adata,
        var_names=topic_prop.columns,
        groupby=groupby,
        figsize=figsize,
        dendrogram=dendrogram,
        swap_axes=swap_axes,
        cmap=cmap,
        save=save,
    )


def trackplot(
    adata,
    topic_prop,
    groupby,
    dendrogram=False,
    swap_axes=True,
    cmap=None,
    figsize=(10, 5),
    save=None,
    **kwargs,
):
    """Creates a trackplot of cell(grouped by groupby) by topics

    Args:
        adata (_type_): Adata object
        topic_prop (_type_): Topic proportions returned by STAMP
        groupby (_type_, optional): Column in adata object to arrange the cells
        figsize (tuple, optional): Figsize. Defaults to (10, 5).
        dendrogram (bool, optional): Whether to cluster. Defaults to False.
        swap_axes (bool, optional): Whether to swap x and y axis. Defaults to True.
        cmap (_type_, optional): What matplotib cmap to use. Defaults to None.
        save (_type_, optional): Whether to save the object. Defaults to None.
    """
    topic_adata = ad.AnnData(topic_prop)
    topic_adata.obs[groupby] = adata.obs[groupby]

    sc.pl.tracksplot(
        adata=topic_adata,
        var_names=topic_prop.columns,
        groupby=groupby,
        figsize=figsize,
        dendrogram=dendrogram,
        swap_axes=swap_axes,
        cmap=cmap,
        save=save,
    )


# def clustermap(
#     beta,
#     gene_dict=None,
#     topn_genes=20,
#     xticklabels="auto",
#     yticklabels="auto",
#     figsize=(10, 6),
#     cmap="viridis",
#     fontsize=5,
#     row_cluster=False,
#     col_cluster=True,
#     standard_scale=0,
#     transpose=False,
#     return_fig=False,
# ):
#     if gene_dict is None:
#         genes = []
#         topics = beta.columns
#         for i in topics:
#             genes.append(beta.nlargest(topn_genes, i).index.tolist())
#         genes = list(set(list(chain.from_iterable(genes))))
#         beta_sub = beta.loc[genes, :]
#     else:
#         genes = [x for x in gene_dict.values()]
#         genes = list(chain.from_iterable(genes))
#         beta_sub = beta.loc[genes, :]

#     if transpose:
#         beta_sub = beta_sub.transpose()

#     fig = sns.clustermap(
#         beta_sub,
#         cmap=cmap,
#         figsize=figsize,
#         row_cluster=row_cluster,
#         col_cluster=col_cluster,
#         standard_scale=standard_scale,
#         xticklabels=xticklabels,
#         yticklabels=yticklabels,
#     )

#     fig.fig.subplots_adjust(right=0.7)
#     fig.ax_cbar.set_position((0.8, 0.4, 0.01, 0.3))

#     fig.ax_heatmap.set_yticklabels(
#         fig.ax_heatmap.get_ymajorticklabels(), fontsize=fontsize, rotation=0
#     )

#     # fig.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#     # fig.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

#     if return_fig:
#         fig = fig.fig
#         return fig


# def heatmap_topic_correlation(
#     topic_prop,
#     spatial_connectivities=None,
#     return_values=False,
#     figsize=(8, 6),
#     cmap="viridis",
#     fontsize=8,
# ):
#     fig, ax = plt.subplots(figsize=figsize)

#     if spatial_connectivities is None:
#         corr = topic_prop.corr()
#         sns.heatmap(
#             corr,
#             annot=True,
#             vmin=-1,
#             vmax=1,
#             cmap=cmap,
#             ax=ax,
#             annot_kws={"fontsize": fontsize},
#             fmt=".2f",
#         )
#     else:
#         spatial_topic_prop = spatial_connectivities @ topic_prop
#         spatial_topic_prop = pd.DataFrame(
#             spatial_topic_prop, index=topic_prop.index, columns=topic_prop.columns
#         )
#         corr = spatial_topic_prop.corr()
#         sns.heatmap(
#             corr,
#             annot=True,
#             vmin=-1,
#             vmax=1,
#             cmap=cmap,
#             ax=ax,
#             annot_kws={"fontsize": fontsize},
#             fmt=".2f",
#         )
#     if return_values:
#         return corr
#     else:
#         return ax


# def enrichment_barplot(
#     enrichments,
#     topic,
#     type="enrichr",
#     figsize=(10, 5),
#     n_enrichments=5,
#     qval_cutoff=0.05,
#     title="auto",
# ):
#     if type == "enrichr":
#         if title == "auto":
#             title = enrichments[topic]["Gene_set"][0]
#         enrichment = enrichments[topic]
#         enrichment = enrichment.loc[enrichment["Adjusted P-value"] < qval_cutoff, :]
#         enrichment = enrichment.sort_values("Adjusted P-value")
#         enrichment = enrichment.iloc[:n_enrichments, :]

#         fig, ax = plt.subplots(figsize=figsize)
#         ax.barh(
#             y=enrichment["Term"],
#             width=-np.log(enrichment["Adjusted P-value"]),
#             fill="blue",
#             align="center",
#         )

#         ax.set_yticklabels(
#             [textwrap.fill(term, 24) for term in enrichment["Term"].values]
#         )

#         ax.set_xlabel("- Log Adjusted P-value")
#         ax.set_title(title)

#         ax.invert_yaxis()

#         plt.tight_layout()
#         return ax

#     if type == "gsea":
#         if title == "auto":
#             title = enrichments[topic]["Name"][0]

#         enrichment = enrichments[topic]
#         enrichment = enrichment.loc[enrichment["NOM p-val"] < qval_cutoff, :]
#         enrichment = enrichment[enrichment["NES"] > 0]
#         enrichment = enrichment.sort_values("NES", ascending=False)
#         enrichment["Term"] = enrichment["Term"].str.replace("_", " ")
#         enrichment = enrichment.iloc[:n_enrichments, :]

#         enrichment["-log_qval"] = -np.log(
#             enrichment["FDR q-val"].astype("float") + 1e-7
#         )

#         fig, ax = plt.subplots(figsize=figsize)

#         ax.barh(y=enrichment["Term"], width=enrichment["NES"], align="center")

#         ax.set_xlabel("NES")
#         ax.set_title(title)

#         ax.set_yticklabels(
#             [textwrap.fill(term, 24) for term in enrichment["Term"].values]
#         )

#         ax.invert_yaxis()
#         plt.tight_layout()

#         return ax


# def enrichment_dotplot(
#     enrichment,
#     type="enrichr",
#     figsize=(10, 5),
#     n_enrichments=10,
#     title="auto",
#     cmap=None,
# ):
#     fig, ax = plt.subplots(figsize=figsize)

#     if type == "enrichr":
#         # enrichment = enrichments[topic].copy()
#         # enrichment = enrichment.loc[enrichment["Adjusted P-value"] < qval_cutoff, :]

#         enrichment["gene_size"] = enrichment["Overlap"].str.split("/").str[1]
#         enrichment["-log_qval"] = -np.log(enrichment["Adjusted P-value"])
#         enrichment["gene_ratio"] = enrichment["Overlap"].str.split("/").str[0].astype(
#             "int"
#         ) / enrichment["Overlap"].str.split("/").str[1].astype("int")

#         if enrichment.shape[0] < n_enrichments:
#             n_enrichments = enrichment.shape[0]

#         enrichment = enrichment.sort_values("gene_ratio")
#         enrichment = enrichment.iloc[:n_enrichments, :]

#         scatter = ax.scatter(
#             x=enrichment["gene_ratio"].values,
#             y=enrichment["Term"].values,
#             s=enrichment["gene_size"].values.astype("int"),
#             c=enrichment["Combined Score"].values,
#             cmap=cmap,
#         )
#         ax.set_xlabel("Gene Ratio")

#         legend1 = ax.legend(
#             *scatter.legend_elements(prop="sizes", num=5),
#             bbox_to_anchor=(1.04, 1),
#             loc="upper left",
#             title="Geneset Size",
#             labelspacing=1,
#             borderpad=1,
#         )
#         ax.legend(
#             *scatter.legend_elements(prop="colors", num=5),
#             bbox_to_anchor=(1.04, 0),
#             loc="lower left",
#             title="Combined Score",
#             labelspacing=1,
#             borderpad=1,
#         )

#         ax.add_artist(legend1)
#         # ax.add_artist(legend2)

#         ax.set_yticklabels(
#             [textwrap.fill(term, 24) for term in enrichment["Term"].values]
#         )

#         if title == "auto":
#             ax.set_title(enrichment["Gene_set"].values[0])

#     if type == "gsea":
#         # enrichment = enrichments[topic].copy()
#         # enrichment = enrichment.loc[enrichment["FDR q-val"] < qval_cutoff, :]

#         enrichment["gene_size"] = enrichment["Tag %"].str.split("/").str[1]
#         enrichment["-log_qval"] = -np.log(
#             enrichment["FDR q-val"].astype("float") + 1e-7
#         )
#         enrichment["gene_ratio"] = enrichment["Tag %"].str.split("/").str[0].astype(
#             "int"
#         ) / enrichment["Tag %"].str.split("/").str[1].astype("int")

#         if enrichment.shape[0] < n_enrichments:
#             n_enrichments = enrichment.shape[0]

#         enrichment = enrichment.sort_values("-log_qval", ascending=False)
#         enrichment = enrichment.iloc[:n_enrichments, :]

#         scatter = ax.scatter(
#             x=enrichment["-log_qval"].values,
#             y=enrichment["Term"].values,
#             s=enrichment["gene_ratio"].values.astype("float"),
#             c=enrichment["NES"].values,
#             cmap=cmap,
#         )
#         ax.set_xlabel("-log q_val")

#         legend1 = ax.legend(
#             *scatter.legend_elements(prop="sizes", num=5),
#             bbox_to_anchor=(1, 1),
#             loc="upper left",
#             title="Gene Ratio",
#             labelspacing=1,
#             borderpad=1,
#         )

#         ax.legend(
#             *scatter.legend_elements(prop="colors", num=5),
#             bbox_to_anchor=(1, 0),
#             loc="lower left",
#             title="NES",
#             labelspacing=1,
#             borderpad=1,
#         )

#         ax.add_artist(legend1)
#         # ax.add_artist(legend2)

#         ax.set_yticklabels(
#             [textwrap.fill(term, 30) for term in enrichment["Term"].values]
#         )

#         if title == "auto":
#             ax.set_title(enrichment["Name"].values[0])

#         ax.invert_yaxis()
#         plt.tight_layout()

#         return ax


def draw_pie(dist, xpos, ypos, size, colors, figsize, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # for incremental pie slices

    cumsum = np.cumsum(dist)
    # normalize
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()
    c = 0
    for r1, r2 in zip(pie[:-1], pie[1:]):
        if r2 - r1 > 0.01:
            angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2, num=100)
            x = [0] + np.cos(angles).tolist() + [0]
            y = [0] + np.sin(angles).tolist() + [0]
            xy = np.column_stack([x, y])
            # print(xy.shape)
            ax.plot(
                [xpos],
                [ypos],
                marker=xy,
                markersize=np.abs(xy).max() * np.array(np.sqrt(size)),
                c=colors[c],
            )
        c += 1

    return ax


def spatialpie(
    adata,
    color,
    N=2,
    *,
    basis: str = "spatial",
    img: Union[np.ndarray, None] = None,
    img_key: Union[str, None, Empty] = _empty,
    library_id: Union[str, Empty] = _empty,
    crop_coord: Tuple[int, int, int, int] = None,
    alpha_img: float = 1.0,
    bw: Optional[bool] = False,
    frameon=False,
    size: float = 1.0,
    scale_factor: Optional[float] = None,
    spot_size: Optional[float] = None,
    na_color: Optional[ColorLike] = None,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    cmap="tab20",
    legend=True,
    figsize=None,
    title=None,
) -> Union[Axes, List[Axes], None]:
    """Creates a spatialpie plot of object. Very ugly according to twitter.

    Args:
        adata (_type_): Adata object
        topic_prop (_type_): Return
        basis (str, optional): Basis to use. Defaults to "spatial".
        img (Union[np.ndarray, None], optional): _description_. Defaults to None.
        img_key (Union[str, None, Empty], optional): _description_. Defaults to _empty.
        library_id (Union[str, Empty], optional): _description_. Defaults to _empty.
        crop_coord (Tuple[int, int, int, int], optional): _description_.
        Defaults to None.
        alpha_img (float, optional): _description_. Defaults to 1.0.
        bw (Optional[bool], optional): _description_. Defaults to False.
        frameon (bool, optional): _description_. Defaults to False.
        size (float, optional): _description_. Defaults to 1.0.
        scale_factor (Optional[float], optional): _description_. Defaults to None.
        spot_size (Optional[float], optional): _description_. Defaults to None.
        na_color (Optional[ColorLike], optional): _description_. Defaults to None.
        show (Optional[bool], optional): _description_. Defaults to None.
        return_fig (Optional[bool], optional): _description_. Defaults to None.
        cmap (str, optional): _description_. Defaults to "tab20".
        legend (bool, optional): _description_. Defaults to True.
        figsize (_type_, optional): _description_. Defaults to None.
        title (_type_, optional): _description_. Defaults to None.

    Returns:
        Union[Axes, List[Axes], None]: _description_
    """
    # get default image params if available
    library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
    img, img_key = _check_img(spatial_data, img, img_key, bw=bw)
    spot_size = _check_spot_size(spatial_data, spot_size)
    scale_factor = _check_scale_factor(
        spatial_data, img_key=img_key, scale_factor=scale_factor
    )
    crop_coord = _check_crop_coord(crop_coord, scale_factor)
    na_color = _check_na_color(na_color, img=img)

    if bw:
        cmap_img = "gray"
    else:
        cmap_img = None

    if scale_factor is not None:
        circle_radius = size * scale_factor * spot_size * 0.5
    else:
        circle_radius = spot_size * 0.5

    if figsize is None:
        figsize = (rcParams["figure.figsize"][0], rcParams["figure.figsize"][1])

    topic_prop = adata.obs[color]
    topic_prop = topic_prop.mask(
        topic_prop.rank(axis=1, method="min", ascending=False) > N, 0
    )
    topic_prop = topic_prop.values

    n_cells = topic_prop.shape[0]
    axs = None
    spatial_coords = adata.obsm[basis]
    n = 0
    colors = plt.get_cmap(cmap)
    colors = colors.colors

    for i in range(n_cells):
        n = n + 1
        if (n % 20000) == 0:
            print(f"{i} number of cells done")
        axs = draw_pie(
            topic_prop[i,],
            xpos=spatial_coords[i, 0] * scale_factor,
            ypos=spatial_coords[i, 1] * scale_factor,
            figsize=figsize,
            size=circle_radius,
            ax=axs,
            colors=colors,
        )

    if legend:
        legend_elements = []
        for i in range(len(colors)):
            legend_elements.append(
                Patch(facecolor=colors[i], edgecolor="w", label=color[i])
            )

        axs.legend(
            handles=legend_elements,
            bbox_to_anchor=(1, 0.5),
            loc="center left",
            labelspacing=0.5,
            borderpad=0.5,
            frameon=frameon,
        )
    if not isinstance(axs, list):
        axs = [axs]
    for ax in axs:
        cur_coords = np.concatenate([ax.get_xlim(), ax.get_ylim()])
        if img is not None:
            ax.imshow(img, cmap=cmap_img, alpha=alpha_img)
        else:
            ax.set_aspect("equal")
            ax.invert_yaxis()
        if crop_coord is not None:
            ax.set_xlim(crop_coord[0], crop_coord[1])
            ax.set_ylim(crop_coord[3], crop_coord[2])
        else:
            ax.set_xlim(cur_coords[0], cur_coords[1])
            ax.set_ylim(cur_coords[3], cur_coords[2])

    ax.set_xlabel(basis + "1")
    ax.set_ylabel(basis + "2")

    ax.set_xticks([])
    ax.set_yticks([])

    if not frameon:
        ax.axis("off")

    if title is not None:
        ax.set_title(title)

    if show is False or return_fig is True:
        return axs


def get_rgb_function(cmap, min_value, max_value):
    r"""Generate a function to map continous values to RGB values using colormap
    between min_value & max_value."""

    if min_value > max_value:
        raise ValueError("Max_value should be greater or than min_value.")

        # if min_value == max_value:
        #     warnings.warn(
        #         "Max_color is equal to min_color. It might be because of the data or
        #  bad
        #         parameter choice. "
        #         "If you are using plot_contours function try increasing
        # max_color_quantile
        #         parameter and"
        #         "removing cell types with all zero values."
        #     )

        def func_equal(x):
            factor = 0 if max_value == 0 else 0.5
            return cmap(np.ones_like(x) * factor)

        return func_equal

    def func(x):
        return cmap(
            (np.clip(x, min_value, max_value) - min_value) / (max_value - min_value)
        )

    return func


def rgb_to_ryb(rgb):
    """
    Converts colours from RGB colorspace to RYB

    Parameters
    ----------

    rgb
        numpy array Nx3

    Returns
    -------
    Numpy array Nx3
    """
    rgb = np.array(rgb)
    if len(rgb.shape) == 1:
        rgb = rgb[np.newaxis, :]

    white = rgb.min(axis=1)
    black = (1 - rgb).min(axis=1)
    rgb = rgb - white[:, np.newaxis]

    yellow = rgb[:, :2].min(axis=1)
    ryb = np.zeros_like(rgb)
    ryb[:, 0] = rgb[:, 0] - yellow
    ryb[:, 1] = (yellow + rgb[:, 1]) / 2
    ryb[:, 2] = (rgb[:, 2] + rgb[:, 1] - yellow) / 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = ryb[mask].max(axis=1) / rgb[mask].max(axis=1)
        ryb[mask] = ryb[mask] / norm[:, np.newaxis]

    return ryb + black[:, np.newaxis]


def ryb_to_rgb(ryb):
    """
    Converts colours from RYB colorspace to RGB

    Parameters
    ----------

    ryb
        numpy array Nx3

    Returns
    -------
    Numpy array Nx3
    """
    ryb = np.array(ryb)
    if len(ryb.shape) == 1:
        ryb = ryb[np.newaxis, :]

    black = ryb.min(axis=1)
    white = (1 - ryb).min(axis=1)
    ryb = ryb - black[:, np.newaxis]

    green = ryb[:, 1:].min(axis=1)
    rgb = np.zeros_like(ryb)
    rgb[:, 0] = ryb[:, 0] + ryb[:, 1] - green
    rgb[:, 1] = green + ryb[:, 1]
    rgb[:, 2] = (ryb[:, 2] - green) * 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = rgb[mask].max(axis=1) / ryb[mask].max(axis=1)
        rgb[mask] = rgb[mask] / norm[:, np.newaxis]

    return rgb + white[:, np.newaxis]


def plot_spatial_general(
    value_df,
    coords,
    labels,
    text=None,
    circle_radius=None,
    display_zeros=False,
    figsize=(10, 10),
    alpha_scaling=1.0,
    max_col=(np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
    max_color_quantile=0.98,
    show_img=True,
    img=None,
    img_alpha=1.0,
    adjust_text=False,
    plt_axis="off",
    axis_y_flipped=False,
    x_y_labels=("", ""),
    crop_x=None,
    crop_y=None,
    text_box_alpha=0.9,
    reorder_cmap=range(7),
    style="fast",
    colorbar_position="right",
    colorbar_label_kw={},
    colorbar_shape={},
    colorbar_tick_size=12,
    colorbar_grid=None,
    image_cmap="Greys_r",
    white_spacing=20,
):
    if value_df.shape[1] > 7:
        raise ValueError(
            "Maximum of 7 cell types / factors can be plotted at the moment"
        )

    def create_colormap(R, G, B):
        spacing = int(white_spacing * 2.55)

        N = 255
        M = 3

        alphas = np.concatenate(
            [[0] * spacing * M, np.linspace(0, 1.0, (N - spacing) * M)]
        )

        vals = np.ones((N * M, 4))
        #         vals[:, 0] = np.linspace(1, R / 255, N * M)
        #         vals[:, 1] = np.linspace(1, G / 255, N * M)
        #         vals[:, 2] = np.linspace(1, B / 255, N * M)
        for i, color in enumerate([R, G, B]):
            vals[:, i] = color / 255
        vals[:, 3] = alphas

        return ListedColormap(vals)

    # Create linearly scaled colormaps
    YellowCM = create_colormap(
        240, 228, 66
    )  # #F0E442 ['#F0E442', '#D55E00', '#56B4E9',
    # '#009E73', '#5A14A5', '#C8C8C8', '#323232']
    RedCM = create_colormap(213, 94, 0)  # #D55E00
    BlueCM = create_colormap(86, 180, 233)  # #56B4E9
    GreenCM = create_colormap(0, 158, 115)  # #009E73
    PinkCM = create_colormap(255, 105, 180)  # #C8C8C8
    WhiteCM = create_colormap(50, 50, 50)  # #323232
    PurpleCM = create_colormap(90, 20, 165)  # #5A14A5
    # LightGreyCM = create_colormap(240, 240, 240)  # Very Light Grey: #F0F0F0

    cmaps = [YellowCM, RedCM, BlueCM, GreenCM, PurpleCM, PinkCM, WhiteCM]

    cmaps = [cmaps[i] for i in reorder_cmap]

    with mpl.style.context(style):
        fig = plt.figure(figsize=figsize)
        if colorbar_position == "right":
            if colorbar_grid is None:
                colorbar_grid = (len(labels), 1)

            shape = {
                "vertical_gaps": 1.5,
                "horizontal_gaps": 0,
                "width": 0.15,
                "height": 0.2,
            }
            shape = {**shape, **colorbar_shape}

            gs = GridSpec(
                nrows=colorbar_grid[0] + 2,
                ncols=colorbar_grid[1] + 1,
                width_ratios=[1, *[shape["width"]] * colorbar_grid[1]],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0], 1],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )
            ax = fig.add_subplot(gs[:, 0], aspect="equal", rasterized=True)

        if colorbar_position == "bottom":
            if colorbar_grid is None:
                if len(labels) <= 3:
                    colorbar_grid = (1, len(labels))
                else:
                    n_rows = round(len(labels) / 3 + 0.5 - 1e-9)
                    colorbar_grid = (n_rows, 3)

            shape = {
                "vertical_gaps": 0.3,
                "horizontal_gaps": 0.6,
                "width": 0.2,
                "height": 0.035,
            }
            shape = {**shape, **colorbar_shape}

            gs = GridSpec(
                nrows=colorbar_grid[0] + 1,
                ncols=colorbar_grid[1] + 2,
                width_ratios=[0.3, *[shape["width"]] * colorbar_grid[1], 0.3],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0]],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )

            ax = fig.add_subplot(gs[0, :], aspect="equal", rasterized=True)

        if colorbar_position is None:
            ax = fig.add_subplot(aspect="equal", rasterized=True)

        if colorbar_position is not None:
            cbar_axes = []
            for row in range(1, colorbar_grid[0] + 1):
                for column in range(1, colorbar_grid[1] + 1):
                    cbar_axes.append(fig.add_subplot(gs[row, column]))

            n_excess = colorbar_grid[0] * colorbar_grid[1] - len(labels)
            if n_excess > 0:
                for i in range(1, n_excess + 1):
                    cbar_axes[-i].set_visible(False)

        ax.set_xlabel(x_y_labels[0])
        ax.set_ylabel(x_y_labels[1])

        if img is not None and show_img:
            ax.imshow(img, alpha=img_alpha, cmap=image_cmap)

        # crop images in needed
        if crop_x is not None:
            ax.set_xlim(crop_x[0], crop_x[1])
        if crop_y is not None:
            ax.set_ylim(crop_y[0], crop_y[1])

        if axis_y_flipped:
            ax.invert_yaxis()

        if plt_axis == "off":
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        counts = value_df.values.copy()

        # plot spots as circles
        c_ord = list(np.arange(0, counts.shape[1]))
        colors = np.zeros((*counts.shape, 4))
        weights = np.zeros(counts.shape)

        for c in c_ord:
            min_color_intensity = counts[:, c].min()
            max_color_intensity = np.min(
                [np.quantile(counts[:, c], max_color_quantile), max_col[c]]
            )

            rgb_function = get_rgb_function(
                cmap=cmaps[c],
                min_value=min_color_intensity,
                max_value=max_color_intensity,
            )

            color = rgb_function(counts[:, c])
            color[:, 3] = color[:, 3] * alpha_scaling

            norm = mpl.colors.Normalize(
                vmin=min_color_intensity, vmax=max_color_intensity
            )

            if colorbar_position is not None:
                cbar_ticks = [
                    min_color_intensity,
                    np.mean([min_color_intensity, max_color_intensity]),
                    max_color_intensity,
                ]
                cbar_ticks = np.array(cbar_ticks)

                if max_color_intensity > 13:
                    cbar_ticks = cbar_ticks.astype(np.int32)
                else:
                    cbar_ticks = cbar_ticks.round(2)

                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmaps[c]),
                    cax=cbar_axes[c],
                    orientation="horizontal",
                    extend="both",
                    ticks=cbar_ticks,
                )

                cbar.ax.tick_params(labelsize=colorbar_tick_size)
                max_color = rgb_function(max_color_intensity / 1.5)
                cbar.ax.set_title(
                    labels[c],
                    **{
                        **{"size": 20, "color": max_color, "alpha": 1},
                        **colorbar_label_kw,
                    },
                )

            colors[:, c] = color
            weights[:, c] = np.clip(counts[:, c] / (max_color_intensity + 1e-10), 0, 1)
            weights[:, c][counts[:, c] < min_color_intensity] = 0

        colors_ryb = np.zeros((*weights.shape, 3))

        for i in range(colors.shape[0]):
            colors_ryb[i] = rgb_to_ryb(colors[i, :, :3])

        def kernel(w):
            return w**2

        kernel_weights = kernel(weights[:, :, np.newaxis])
        weighted_colors_ryb = (colors_ryb * kernel_weights).sum(
            axis=1
        ) / kernel_weights.sum(axis=1)
        weighted_colors = np.zeros((weights.shape[0], 4))
        weighted_colors[:, :3] = ryb_to_rgb(weighted_colors_ryb)
        weighted_colors[:, 3] = colors[:, :, 3].max(axis=1)

        if display_zeros:
            weighted_colors[weighted_colors[:, 3] == 0] = [
                210 / 255,
                210 / 255,
                210 / 255,
                1,
            ]

        ax.scatter(
            x=coords[:, 0], y=coords[:, 1], c=weighted_colors, s=circle_radius**2
        )

        # size in circles is radius
        # add text
        if text is not None:
            bbox_props = dict(boxstyle="round", ec="0.5", alpha=text_box_alpha, fc="w")
            texts = []
            for x, y, s in zip(
                np.array(text.iloc[:, 0].values).flatten(),
                np.array(text.iloc[:, 1].values).flatten(),
                text.iloc[:, 2].tolist(),
            ):
                texts.append(
                    ax.text(x, y, s, ha="center", va="bottom", bbox=bbox_props)
                )

            if adjust_text:
                from adjustText import adjust_text

                adjust_text(texts, arrowprops=dict(arrowstyle="->", color="w", lw=0.5))

    plt.grid(False)
    return fig


def plot_spatial(
    adata,
    topic_prop,
    basis="spatial",
    bw=False,
    img=None,
    library_id=_empty,
    crop_coord=None,
    img_key=_empty,
    spot_size=None,
    na_color=None,
    scale_factor=None,
    scale_default=0.5,
    show_img=True,
    display_zeros=False,
    figsize=(10, 10),
    **kwargs,
):
    """Plot taken from cell2location at https://github.com/BayraktarLab/cell2location.
    Able to display zeros and also on umap through the basis function

    Args:
        adata (_type_): Adata object with spatial coordinates in adata.obsm['spatial']
        topic_prop (_type_): Topic proportion obtained from STAMP.
        basis (str, optional): Which basis to plot in adata.obsm. Defaults to "spatial".
        bw (bool, optional): Defaults to False.
        img (_type_, optional): . Defaults to None.
        library_id (_type_, optional): _description_. Defaults to _empty.
        crop_coord (_type_, optional): _description_. Defaults to None.
        img_key (_type_, optional): _description_. Defaults to _empty.
        spot_size (_type_, optional): _description_. Defaults to None.
        na_color (_type_, optional): _description_. Defaults to None.
        scale_factor (_type_, optional): _description_. Defaults to None.
        scale_default (float, optional): _description_. Defaults to 0.5.
        show_img (bool, optional): Whether to display spatial image. Sets to false
        automatically when displaying umap. Defaults to True.
        display_zeros (bool, optional): Whether to display cells that have low counts
        values to grey colour. Defaults to False.
        figsize (tuple, optional): Figsize of image. Defaults to (10, 10).

    Returns:
        _type_: Function taken from cell2location at
        https://cell2location.readthedocs.io/en/latest/_modules/cell2location/plt/plot_spatial.html#plot_spatial.
        Able to plot both on spatial and umap coordinates. Still very raw.
    """
    # get default image params if available
    library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
    img, img_key = _check_img(spatial_data, img, img_key, bw=bw)
    spot_size = _check_spot_size(spatial_data, spot_size)
    scale_factor = _check_scale_factor(
        spatial_data, img_key=img_key, scale_factor=scale_factor
    )
    crop_coord = _check_crop_coord(crop_coord, scale_factor)
    na_color = _check_na_color(na_color, img=img)

    if scale_factor is not None:
        circle_radius = scale_factor * spot_size * 0.5 * scale_default
    else:
        circle_radius = spot_size * 0.5

    if show_img is True:
        kwargs["show_img"] = True
        kwargs["img"] = img

    kwargs["coords"] = adata.obsm[basis] * scale_factor

    fig = plot_spatial_general(
        value_df=topic_prop,
        labels=topic_prop.columns,
        circle_radius=circle_radius,
        figsize=figsize,
        display_zeros=display_zeros,
        **kwargs,
    )  # cell abundance values
    plt.gca().invert_yaxis()

    return fig


def spatial(
    adata,
    color=None,
    cmap=None,
    frameon=None,
    title=None,
    wspace=None,
    hspace=0.25,
    palette=None,
    colorbar_loc="right",
    size=1,
    basis="spatial",
    vmax=None,
    ncols=4,
    layer=None,
    show=True,
    *args,
    **kwargs,
):
    """A faster simple function that uses sc.pl.embedding to plot for non-visium data
    so it dont take too long. ~sleep. Very inflexible.

    Args:
        adata (_type_): Annotated data matrix.
        color (_type_): Keys for annotations of observations/cells or variables/genes
        size (int, optional): size of spots. Defaults to 1.
        basis (str, optional): basis in obsm. Defaults to "spatial".
        vmax (str, optional): The value representing the upper limit of the color scale. Defaults to "p99".
        show (bool, optional): Show the plot, do not return axis. Defaults to True.

    Returns:
        _type_: A plot
    """
    ax = sc.pl.embedding(
        adata,
        basis=basis,
        show=False,
        color=color,
        wspace=wspace,
        hspace=hspace,
        palette=palette,
        vmax=vmax,
        size=size,
        ncols=ncols,
        cmap=cmap,
        frameon=frameon,
        colorbar_loc=colorbar_loc,
        title=title,
        layer=layer,
        *args,
        **kwargs,
    )
    if isinstance(ax, list):
        [axs.invert_yaxis() for axs in ax]
        [axs.set_aspect("equal") for axs in ax]
    else:
        ax.invert_yaxis()
        ax.set_aspect("equal")
    if show is False:
        return ax
