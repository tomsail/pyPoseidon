"""
gmsh module

"""
# Copyright 2018 European Union
# This file is part of pyposeidon.
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by the European Commission - subsequent versions of the EUPL (the "Licence").
# Unless required by applicable law or agreed to in writing, software distributed under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and limitations under the Licence.

import pandas as pd
import numpy as np
import geopandas as gp
import xarray as xr
import os
from tqdm.auto import tqdm
import sys
import oceanmesh as om
import subprocess
import shapely
import shlex

from pyposeidon.utils.spline import use_spline
from pyposeidon.utils.global_bgmesh import make_bgmesh_global
from pyposeidon.boundary import tag
from pyposeidon.utils.stereo import to_stereo, to_3d, to_lat_lon
from pyposeidon.utils.pos import to_st, to_global_pos, to_sq
from pyposeidon.utils.scale import scale_dem
from pyposeidon.utils.topology import (
    MakeTriangleFaces_periodic,
    MakeQuadFaces,
    tria_to_df,
    tria_to_df_3d,
    quads_to_df,
)
import pyposeidon.dem as pdem
from pyposeidon.tools import orient

import multiprocessing

NCORES = max(1, min(multiprocessing.cpu_count() - 1, 20))

from joblib import Parallel, delayed, parallel_backend

import logging

logger = logging.getLogger(__name__)


def get_ibounds(df, mm):
    if df.shape[0] == 0:
        disable = True
    else:
        disable = False

    ibounds = []

    for row in tqdm(df.itertuples(index=True, name="Pandas"), total=df.shape[0], disable=disable):
        inodes, xyz = mm.getNodesForPhysicalGroup(dim=getattr(row, "dim"), tag=getattr(row, "tag"))
        db = pd.DataFrame({"node": inodes - 1})
        db["type"] = "island"
        db["id"] = -(getattr(row, "Index") + 1)

        ibounds.append(db)

    return ibounds


def check_mesh(ds, stereo_to_ll=False):
    """
    Check the mesh and reverse any counter-clockwise (CW) triangles.

    If the determinant of the cross product of two edges of a triangle is negative,
    the triangle is CW. This function reverses the triangle if it is CW.

    Additionally, this function removes any flat (degenerate) triangles from the mesh.
    A triangle is considered flat if its area is less than a threshold value.

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    ds : xr.Dataset

    TODO: MOVE THIS FUNCTION IN A COMMON SPACE (FOR OTHER MESHERS)

    """
    logger.info("checking mesh..\n")

    tris = ds.SCHISM_hgrid_face_nodes.data
    x = ds.SCHISM_hgrid_node_x.data
    y = ds.SCHISM_hgrid_node_y.data

    t12 = -x[tris[:, 0]] + x[tris[:, 1]]
    t13 = -x[tris[:, 0]] + x[tris[:, 2]]
    t22 = -y[tris[:, 0]] + y[tris[:, 1]]
    t23 = -y[tris[:, 0]] + y[tris[:, 2]]
    #
    det = t12 * t23 - t22 * t13  # as defined in GEOELT (sources/utils/geoelt.f)
    #
    ccw_ = det > 0
    cw_ = det < 0
    flat_ = abs(det) <= 10e-6

    # Reverse CW triangles
    ds.SCHISM_hgrid_face_nodes[~ccw_] = ds.SCHISM_hgrid_face_nodes[~ccw_][:, ::-1]
    if cw_.sum() > 0:
        logger.info(" > reversed " + str(cw_.sum()) + " CW triangles")

    if flat_.sum() > 0:
        non_flat_tris = tris[~flat_]
        if non_flat_tris.size > 0:  # Check if non_flat_tris is not empty
            nodes_ = np.unique(non_flat_tris)

            # Create a mapping from old indices to new indices
            map_old_new_ = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_)}

            # Apply the mapping to update the indices in non_flat_tris
            non_flat_tris_mapped = np.vectorize(map_old_new_.get)(non_flat_tris)

            x_ = ds.SCHISM_hgrid_node_x.values[nodes_]
            y_ = ds.SCHISM_hgrid_node_y.values[nodes_]

            # using oceanmesh to cleanup and fix the mesh
            points, cells = om.make_mesh_boundaries_traversable(np.column_stack((x_, y_)), non_flat_tris_mapped)
            points, cells = om.delete_faces_connected_to_one_face(points, cells)

            ds = om_to_xarray(points, cells, stereo_to_ll=stereo_to_ll)
            logger.info(" > removed " + str(flat_.sum()) + " flat triangles")
            logger.info(f" > Filtered {len(ds.node.values) - len(ds.node.values)} boundary nodes")
            logger.info(f" > removed {len(tris) - len(cells)} elements in total.\n")
    else:
        logger.info("No flat triangles detected, no changes made.")

    return ds


def get(contours, **kwargs):
    """
    Create a `oceanmesh` mesh.

    !!! danger ""
        Due to a limitation of the Library rendering the docstrings, all arguments are marked
        as `required`, nevertheless they are all `Optional`.

    Args:
        contours GeoDataFrame: Provide boundaries and metadata.
        rpath str: Path for output. Defaults to `"."`.
        use_bindings bool: Flag for using python API as opposed to binary. Defaults to `True`.
        dem_source str: Path or url to bathymetric data.
        bgmesh str: Path to a mesh scale file. Defaults to `None`.
        setup_only bool: Flag for setup only (no execution). Defaults to `False`.
    """

    logger.info("Creating mesh with Oceanmesh\n")

    rpath = kwargs.get("rpath", ".")

    if not os.path.exists(rpath):
        os.makedirs(rpath)

    gpath = os.path.join(rpath, "oceanmesh")
    if not os.path.exists(gpath):
        os.makedirs(gpath)

    use_bindings = kwargs.get("use_bindings", True)
    setup_only = kwargs.get("setup_only", False)
    bgmesh = kwargs.get("bgmesh", None)

    if bgmesh is None:
        dem_source = kwargs.get("dem_source", None)
        if dem_source:
            bgmesh = "auto"
            kwargs.update({"bgmesh": "auto"})

    gglobal = kwargs.get("gglobal", False)

    if bgmesh == "auto":
        try:
            rpath = kwargs.get("rpath", ".")

            if not os.path.exists(rpath + "/oceanmesh/"):  # check if run folder exists
                os.makedirs(rpath + "/oceanmesh/")

            fnc = rpath + "/oceanmesh/bgmesh.nc"
            if gglobal:
                dem = pdem.Dem(**kwargs)
                nds, lms = make_bgmesh_global(contours, fnc, dem.Dataset, **kwargs)
                dh = to_global_pos(nds, lms, fnc, **kwargs)
            else:
                dh = make_bgmesh(contours, fnc, **kwargs)

            kwargs.update({"bgmesh": fnc})

        except OSError as e:
            logger.warning("bgmesh failed... continuing without background mesh size")
            dh = None
            kwargs.update({"bgmesh": None})

    if gglobal:
        gr = make_oceanmesh_global(contours, **kwargs)
    else:
        gr = make_oceanmesh(contours, **kwargs)

    try:
        bg = dh
    except:
        bg = None

    return gr, bg


def gset(df, **kwargs):
    logger.info("Interpolating coastal points")

    lc = kwargs.get("lc", 0.5)

    df["lc"] = lc
    df = df.apply(pd.to_numeric)

    # Resample to equidistant points

    conts = np.unique(df.index[df.tag < 0].get_level_values(0))
    conts = [x for x in conts if x not in ["line0"]]  # except the outer LineString

    ibs = len(conts)

    ndfsfs = {}
    for ic in tqdm(range(ibs)):
        contour = conts[ic]
        curve = df.loc[contour, ["lon", "lat"]]
        curve = pd.concat([curve, curve.loc[0:0]]).reset_index(drop=True)
        di = use_spline(curve, ds=0.01, method="slinear")
        di["z"] = df.loc[contour].z.values[0]
        di["tag"] = df.loc[contour].tag.values[0].astype(int)
        di["lc"] = df.loc[contour].lc.values[0]
        ndfsfs.update({contour: di.drop_duplicates(["lon", "lat"])})

    df_ = pd.concat(ndfsfs, axis=0)
    df_["z"] = df_.z.values.astype(int)
    df_["tag"] = df_.tag.values.astype(int)

    # Line0

    logger.info("Setting outermost boundary")

    df0 = df.loc["line0"]

    mtag = df0.tag.min()
    mtag = mtag.astype(int)

    nd0 = {}
    for ic in tqdm(range(mtag, 0)):
        contour = df0.tag == ic
        curve = df0.loc[contour, ["lon", "lat"]].reset_index(drop=True)
        #    curve = pd.concat([curve,curve.loc[0:0]]).reset_index(drop=True)
        di = use_spline(curve, ds=0.01, method="slinear")
        di["z"] = df0.loc[contour].z.values[0]
        di["tag"] = df0.loc[contour].tag.values[0].astype(int)
        di["lc"] = df0.loc[contour].lc.values[0]
        nd0.update({ic: di.drop_duplicates(["lon", "lat"])})

    # Join Line0
    df0_ = df0.copy()
    for l in range(mtag, 0):
        idx = df0_.loc[df0_.tag == l].index
        df0_ = pd.concat([df0_.iloc[: idx[0]], nd0[l], df0_.iloc[idx[-1] + 1 :]])
        df0_.reset_index(drop=True, inplace=True)

    df0_ = pd.concat({"line0": df0_})

    # join all
    ddf = pd.concat([df0_, df_])

    ddf["z"] = ddf.z.values.astype(int)
    ddf["tag"] = ddf.tag.values.astype(int)

    # check orientation
    r0 = ddf.loc["line0"]

    if not shapely.geometry.LinearRing(r0[["lon", "lat"]].values).is_ccw:
        rf0 = ddf.loc["line0"].iloc[::-1].reset_index(drop=True)
        ddf.loc["line0"] = rf0.values

    ddf = ddf.apply(pd.to_numeric)

    return ddf


def outer_boundary(df, **kwargs):
    lc = kwargs.get("lc", 0.5)

    df_ = df.loc[df.tag != "island"].reset_index(drop=True)  # all external contours

    # store xy in a DataFrame
    dic = {}
    for k, d in df_.iterrows():
        out = pd.DataFrame(d.geometry.coords[:], columns=["lon", "lat"])
        out["lindex"] = d.lindex
        out = out.drop_duplicates(["lon", "lat"])
        if d.tag == "land":  # drop end points in favor or open tag
            out = out[1:-1]
        dic.update({"line{}".format(k): out})
    o1 = pd.concat(dic, axis=0).droplevel(0).reset_index(drop=True)
    o1 = o1.drop_duplicates(["lon", "lat"])

    # Do linemerge of outer contours
    lss = df_.geometry.values
    merged = shapely.ops.linemerge(list(lss))
    o2 = pd.DataFrame({"lon": merged.xy[0], "lat": merged.xy[1]})  # convert to DataFrame
    o2 = o2.drop_duplicates()

    rb0 = o2.merge(o1)  # merge to transfer the lindex

    rb0["z"] = 0

    # check orientation
    if not shapely.geometry.LinearRing(rb0[["lon", "lat"]].values).is_ccw:
        rb0_ = rb0.iloc[::-1].reset_index(drop=True)
        rb0 = rb0_

    rb0["lc"] = lc

    logger.info("Computing edges")
    edges = [list(a) for a in zip(np.arange(rb0.shape[0]), np.arange(rb0.shape[0]) + 1, rb0.lindex.values)]  # outer
    edges[-1][1] = 0

    # sort (duplicated bounds)
    edges = pd.DataFrame(edges, columns=["index", "ie", "lindex"])

    edges["lindex1"] = pd.concat([rb0.loc[1:, "lindex"], rb0.loc[0:0, "lindex"]]).reset_index(drop=True).astype(int)
    edges["que"] = np.where(
        ((edges["lindex"] != edges["lindex1"]) & (edges["lindex"] > 0) & (edges["lindex"] < 1000)),
        edges["lindex1"],
        edges["lindex"],
    )

    edges = edges.reset_index().loc[:, ["index", "ie", "que"]]

    rb0["bounds"] = edges.loc[:, ["index", "ie"]].values.tolist()

    # get boundary types
    land_lines = {
        your_key: edges.loc[edges.que == your_key].index.values
        for your_key in [x for x in edges.que.unique() if x > 1000]
    }
    open_lines = {
        your_key: edges.loc[edges.que == your_key].index.values
        for your_key in [x for x in edges.que.unique() if x < 1000]
    }

    return rb0, land_lines, open_lines


def make_bgmesh(df, fnc, **kwargs):
    lon_min = df.bounds.minx.min()
    lon_max = df.bounds.maxx.max()
    lat_min = df.bounds.miny.min()
    lat_max = df.bounds.maxy.max()

    kwargs_ = kwargs.copy()
    kwargs_.pop("lon_min", None)
    kwargs_.pop("lon_max", None)
    kwargs_.pop("lat_min", None)
    kwargs_.pop("lat_max", None)

    dem = kwargs.get("dem_source", None)

    if not isinstance(dem, xr.Dataset):
        logger.info("Reading DEM")
        dem = pdem.Dem(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, **kwargs_)
        dem = dem.Dataset

    res_min = kwargs.get("resolution_min", 0.01)
    res_max = kwargs.get("resolution_max", 0.5)

    logger.info("Evaluating bgmesh")

    # scale bathymetry
    try:
        b = dem.adjusted.to_dataframe()
    except:
        b = dem.elevation.to_dataframe()

    b = b.reset_index()

    b.columns = ["latitude", "longitude", "z"]

    nodes = scale_dem(b, res_min, res_max, **kwargs)

    x = dem.longitude.values
    y = dem.latitude.values

    quad = MakeQuadFaces(y.shape[0], x.shape[0])
    elems = pd.DataFrame(quad, columns=["a", "b", "c", "d"])
    df = quads_to_df(elems, nodes)

    dh = xr.Dataset(
        {
            "h": (
                ["latitude", "longitude"],
                nodes.d2.values.flatten().reshape((y.shape[0], x.shape[0])),
            )
        },
        coords={"longitude": ("longitude", x), "latitude": ("latitude", y)},
    )

    logger.info("Saving bgmesh to {}".format(fnc))
    dh.attrs["crs"] = "EPSG:4326"  # EPSG:4326 is the code for WGS84
    dh.to_netcdf(fnc)  # save bgmesh

    kwargs.update({"bgmesh": fnc})

    return dh


def make_oceanmesh(df, **kwargs):
    logger.info("Executing oceanmesh")

    # get options
    interpolate = kwargs.get("interpolate", False)
    espg = kwargs.pop("ESPG", 4326)
    lc = kwargs.get("lc", 0.5)
    bgmesh = kwargs.get("bgmesh")
    res_min = kwargs.pop("resolution_min", 0.01)
    res_max = kwargs.pop("resolution_max", 0.5)
    rpath = kwargs.get("rpath", ".")
    dem_source = kwargs.get("dem_source", None)
    bathy_gradient = kwargs.get("bathy_gradient", True)
    grad = kwargs.get("gradiation", 0.11)
    iter = kwargs.get("iterations", 50)
    pfix = kwargs.get("pfix", None)
    alpha_slp = kwargs.get("alpha_slope", 30)
    alpha_wl = kwargs.get("alpha_wavelength", 30)
    plot = kwargs.get("plot", False)
    #
    if interpolate:  # coastlines
        df = gset(df, **kwargs)
    else:
        df = df
    # Create a Shoreline instance
    lon_min = df.bounds.minx.min()
    lon_max = df.bounds.maxx.max()
    lat_min = df.bounds.miny.min()
    lat_max = df.bounds.maxy.max()
    #
    # set ESPG
    crs = f"EPSG:{espg}"
    # storing input data
    df_ = df.loc[df.tag != "open"].reset_index(drop=True)  # all coasts
    fshp = rpath + "/oceanmesh/" + "om_input.shp"
    gdf = gp.GeoDataFrame(df_, geometry="geometry")
    gdf.set_crs(crs, inplace=True)
    gdf.to_file(fshp, driver="ESRI Shapefile")
    #
    extent = om.Region(extent=(lon_min, lon_max, lat_min, lat_max), crs=crs)
    shoreline = om.Shoreline(shp=fshp, bbox=extent.bbox, h0=res_min, crs=crs)
    domain = om.signed_distance_function(shoreline)
    #
    if bgmesh is None or bgmesh == "om":
        h_funs = []
        logger.info("oceanmesh: distance function")
        h_funs.append(om.distance_sizing_function(shoreline, max_edge_length=res_max, rate=grad))
        logger.info("oceanmesh: feature size function")
        h_funs.append(
            om.feature_sizing_function(
                shoreline,
                domain,
                min_edge_length=res_min,
                max_edge_length=res_max,
                crs=crs,
            )
        )
        if dem_source:
            logger.info(dem_source)
            dem = om.DEM(dem_source, crs=crs, bbox=extent)
            logger.info("oceanmesh: WL size function")
            h_funs.append(
                om.wavelength_sizing_function(
                    dem, wl=alpha_wl, period=12.42 * 3600, min_edgelength=res_min, max_edge_length=res_max, crs=crs
                )  # use the M2-tide period (in seconds)
            )
            if bathy_gradient:
                logger.info("oceanmesh: bathymetric size function")
                h_funs.append(
                    om.bathymetric_gradient_sizing_function(
                        dem,
                        slope_parameter=alpha_slp,
                        min_edge_length=res_min,
                        max_edge_length=res_max,
                        crs=crs,
                    )
                )
        edge_length = om.compute_minimum(h_funs)
        edge_length = om.enforce_mesh_gradation(edge_length, gradation=grad)
    else:
        dh = xr.open_dataset(bgmesh)
        #
        x = dh.longitude.values
        y = dh.latitude.values
        #
        dx = np.diff(x)
        dy = np.diff(y)
        dx_rounded = np.round(dx, 6)
        dy_rounded = np.round(dy, 6)
        unique_dx = np.unique(dx_rounded)
        unique_dy = np.unique(dy_rounded)
        #
        lon_min = x.min()
        lon_max = x.max()
        lat_min = y.min()
        lat_max = y.max()
        #
        extent2 = om.Region(extent=(lon_min, lon_max, lat_min, lat_max), crs=crs)
        edge_length = om.Grid(
            bbox=extent2.bbox,
            dx=unique_dx,
            dy=unique_dy,
            extrapolate=True,
            values=dh.h.values.T,
            crs=crs,
            hmin=res_min,
        )

        if res_max is not None:
            edge_length.values[edge_length.values > res_max] = res_max

        edge_length.values[edge_length.values < res_min] = res_min

        edge_length.build_interpolant()
        edge_length = om.enforce_mesh_gradation(edge_length, gradation=grad)

    edge_length.values = np.abs(edge_length.values)
    if plot:
        fig, ax, pc = edge_length.plot(holding=True, plot_colorbar=True)
        shoreline.plot(ax=ax)
    logger.info("oceanmesh: generate mesh")
    points, cells = om.generate_mesh(domain, edge_length, max_iter=iter, pfix=pfix)

    # remove degenerate mesh faces and other common problems in the mesh
    points, cells = om.make_mesh_boundaries_traversable(points, cells)
    points, cells = om.delete_faces_connected_to_one_face(points, cells)

    # apply a Laplacian smoother
    points, cells = om.laplacian2(points, cells, max_iter=iter)
    nodes = pd.DataFrame(
        data={
            "x": points[:, 0],
            "y": points[:, 1],
            "z": np.zeros(len(points[:, 0])),
        }
    )
    tria = pd.DataFrame(
        data={
            "a": cells[:, 0],
            "b": cells[:, 1],
            "c": cells[:, 2],
        }
    )
    nodes = nodes.apply(pd.to_numeric)
    tria = tria.apply(pd.to_numeric)

    # boundaries
    logger.info("oceanmesh: boundaries")
    df_open = df[df["tag"] == "open"]
    opengp = gp.GeoDataFrame(df_open, geometry="geometry")
    openb = []
    islb = []
    bounds = om.edges.get_boundary_edges(cells)

    for bnd in bounds:
        p1 = shapely.Point(points[bnd[0]])
        p2 = shapely.Point(points[bnd[1]])
        dist1 = opengp.distance(p1).min()
        dist2 = opengp.distance(p2).min()
        if (dist1 < res_min) and (dist2 < res_min):
            openb.append(bnd)
        else:
            islb.append(bnd)

    # open boundaries
    if len(openb) > 0:
        odf = pd.DataFrame({"node": np.unique(np.array(openb).flatten())})
        odf["type"] = "open"
        odf["id"] = 1
    else:
        odf = None

    # island boundaries
    if len(islb) > 0:
        idf = pd.DataFrame({"node": np.unique(np.array(islb).flatten())})
        idf["type"] = "island"
        idf["id"] = -1
    else:
        idf = None

    tbf = pd.concat([odf, idf])
    tbf = tbf.reset_index(drop=True)
    tbf.index.name = "bnodes"

    els = xr.DataArray(
        tria.loc[:, ["a", "b", "c"]].values,
        dims=["nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"],
        name="SCHISM_hgrid_face_nodes",
    )

    nod = (
        nodes.loc[:, ["x", "y"]]
        .to_xarray()
        .rename(
            {
                "index": "nSCHISM_hgrid_node",
                "x": "SCHISM_hgrid_node_x",
                "y": "SCHISM_hgrid_node_y",
            }
        )
    )
    nod = nod.drop_vars("nSCHISM_hgrid_node")

    dep = xr.Dataset({"depth": (["nSCHISM_hgrid_node"], np.zeros(nod.nSCHISM_hgrid_node.shape[0]))})

    gr = xr.merge([nod, dep, els, tbf.to_xarray()])  # total

    return gr


def om_to_xarray(points, cells, stereo_to_ll=True):
    nodes = pd.DataFrame(
        data={
            "x": points[:, 0],
            "y": points[:, 1],
            "z": np.zeros(len(points[:, 0])),
        }
    )
    tria = pd.DataFrame(
        data={
            "a": cells[:, 0],
            "b": cells[:, 1],
            "c": cells[:, 2],
        }
    )
    nodes = nodes.apply(pd.to_numeric)
    tria = tria.apply(pd.to_numeric)

    # boundaries (all are islands)
    logger.info("oceanmesh: boundaries")

    bounds = om.edges.get_boundary_edges(cells)
    if len(bounds) > 0:
        tbf = pd.DataFrame({"node": np.unique(np.array(bounds).flatten())})
        tbf["type"] = "island"
        tbf["id"] = -1
    else:
        tbf = None

    tbf = tbf.reset_index(drop=True)
    tbf.index.name = "bnodes"

    # convert to lat/lon
    if stereo_to_ll:
        if nodes.z.any() != 0:
            xd, yd = to_lat_lon(nodes.x, nodes.y, nodes.z)
            nodes["x"] = xd
            nodes["y"] = yd
        else:
            xd, yd = to_lat_lon(nodes.x, nodes.y)
            nodes["x"] = xd
            nodes["y"] = yd
    else:
        nodes["x"] = nodes.x
        nodes["y"] = nodes.y

    els = xr.DataArray(
        tria.loc[:, ["a", "b", "c"]].values,
        dims=["nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"],
        name="SCHISM_hgrid_face_nodes",
    )

    nod = (
        nodes.loc[:, ["x", "y"]]
        .to_xarray()
        .rename(
            {
                "index": "nSCHISM_hgrid_node",
                "x": "SCHISM_hgrid_node_x",
                "y": "SCHISM_hgrid_node_y",
            }
        )
    )
    nod = nod.drop_vars("nSCHISM_hgrid_node")

    dep = xr.Dataset({"depth": (["nSCHISM_hgrid_node"], np.zeros(nod.nSCHISM_hgrid_node.shape[0]))})

    return xr.merge([nod, dep, els, tbf.to_xarray()])  # total


def make_oceanmesh_global(df, **kwargs):
    logger.info("Executing oceanmesh")
    import matplotlib.pyplot as plt

    # get options
    interpolate = kwargs.get("interpolate", False)
    bgmesh = kwargs.get("bgmesh")
    res_min = kwargs.pop("resolution_min", 0.1)
    res_max = kwargs.pop("resolution_max", 1)
    rpath = kwargs.get("rpath", ".")
    dem_source = kwargs.get("dem_source", None)
    bathy_gradient = kwargs.get("bathy_gradient", True)
    grad = kwargs.get("gradiation", 0.11)
    iter = kwargs.get("iterations", 50)
    tg_database = kwargs.get("obs", None)
    alpha_slp = kwargs.get("alpha_slope", 30)
    alpha_wl = kwargs.get("alpha_wavelength", 30)
    plot = kwargs.get("plot", False)

    # add fixed points in the mesh (stations gauges)
    if tg_database:
        logger.info("get stations from {}\n".format(tg_database))
        tg = gp.read_file(tg_database)
        tg = tg.rename(columns={"latitude": "lat", "longitude": "lon"})  # to make sure we have lat/lon fields
        # Convert the 'lat' and 'lon' columns to float
        tg["lat"] = tg["lat"].astype(float)
        tg["lon"] = tg["lon"].astype(float)
    else:
        tg = None
    if tg is not None:
        p = np.array([tg.lon.values, tg.lat.values])
        x, y = to_stereo(p[0, :], p[1, :])
        pfix = np.asarray([x, y]).T
    else:
        pfix = np.array([])
        pfix = pfix.reshape(-1, 2)
    #
    if interpolate:  # coastlines
        df = gset(df, **kwargs)
    else:
        df = df

    # set ESPG
    EPSG = 4326  # hardcoded
    bbox = (-180.00, 180.00, -89.00, 90.00)  # hardcoded
    crs = f"EPSG:{EPSG}"
    # storing input data
    fshp_ste = rpath + "/oceanmesh/om_stereo.shp"
    df.set_crs(epsg=4326, inplace=True)
    df.to_file(fshp_ste, driver="ESRI Shapefile")
    fshp_ll = rpath + "/oceanmesh/coasts.shp"

    extent = om.Region(extent=bbox, crs=crs)
    shoreline = om.Shoreline(shp=fshp_ll, bbox=extent.bbox, h0=res_min, crs=crs)
    sdf = om.signed_distance_function(shoreline)

    if bgmesh is None or bgmesh == "om":  # use oceanmesh
        h_funs = []
        logger.info("oceanmesh: distance function")
        h_funs.append(om.distance_sizing_function(shoreline, rate=grad))
        logger.info("oceanmesh: feature size function")
        h_funs.append(
            om.feature_sizing_function(
                shoreline,
                sdf,
                min_edge_length=res_min,
                max_edge_length=res_max,
                crs=EPSG,
            )
        )
        if dem_source:
            dem = om.DEM(dem_source, crs=crs, bbox=extent)
            logger.info("oceanmesh: WL size function")
            h_funs.append(
                om.wavelength_sizing_function(
                    dem, wl=alpha_wl, period=12.42 * 3600, min_edgelength=res_min, max_edge_length=res_max, crs=crs
                )  # use the M2-tide period (in seconds)
            )
            if bathy_gradient:
                logger.info("oceanmesh: bathymetric size function")
                h_funs.append(
                    om.bathymetric_gradient_sizing_function(
                        dem,
                        slope_parameter=alpha_slp,
                        min_edge_length=res_min,
                        max_edge_length=res_max,
                        crs=crs,
                    )
                )
        edge_length = om.compute_minimum(h_funs)
        if plot:
            fig, ax, pc = edge_length.plot(holding=True, plot_colorbar=True)
            shoreline.plot(ax=ax)
        edge_length = om.enforce_mesh_gradation(edge_length, gradation=grad, stereo=False)
        edge_length.values = np.abs(edge_length.values)
        if plot:
            fig, ax, pc = edge_length.plot(holding=True, plot_colorbar=True)
            shoreline.plot(ax=ax)

        # stereo shoreline
        shoreline_stereo = om.Shoreline(shp=fshp_ste, bbox=extent.bbox, h0=res_min, crs=crs, stereo=True)
        domain = om.signed_distance_function(shoreline_stereo)
        # add north pole in the mesh to avoid triangle over the north pole
        if domain.eval([[0, 0]]):
            pfix = np.append(pfix, [[0, 0]], axis=0)
    else:
        raise ValueError("Not implemented: bgmesh must be 'om' or None")
        dh = xr.open_dataset(bgmesh)
        #
        x = dh.longitude.values
        y = dh.latitude.values
        #
        dx = np.diff(x)
        dy = np.diff(y)
        dx_rounded = np.round(dx, 6)
        dy_rounded = np.round(dy, 6)
        unique_dx = np.unique(dx_rounded)
        unique_dy = np.unique(dy_rounded)
        #
        lon_min = x.min()
        lon_max = x.max()
        lat_min = y.min()
        lat_max = y.max()
        #
        extent2 = om.Region(extent=(lon_min, lon_max, lat_min, lat_max), crs=crs)
        edge_length = om.Grid(
            bbox=extent2.bbox,
            dx=unique_dx,
            dy=unique_dy,
            extrapolate=True,
            values=dh.h.values.T,
            crs=crs,
            hmin=res_min,
        )

        if res_max is not None:
            edge_length.values[edge_length.values > res_max] = res_max

        edge_length.values[edge_length.values < res_min] = res_min

        edge_length.build_interpolant()
        edge_length = om.enforce_mesh_gradation(edge_length, gradation=grad)

    logger.info("oceanmesh: generate mesh.. ")

    points, cells = om.generate_mesh(domain, edge_length, stereo=True, max_iter=iter, pfix=pfix)

    # remove degenerate mesh faces and other common problems in the mesh
    points, cells = om.make_mesh_boundaries_traversable(points, cells)
    points, cells = om.delete_faces_connected_to_one_face(points, cells)

    # apply a Laplacian smoother
    logger.info("oceanmesh: apply Laplacian smoother")
    points, cells = om.laplacian2(points, cells, max_iter=iter, pfix=pfix)

    gr = om_to_xarray(points, cells)
    gr = check_mesh(gr)

    return gr
