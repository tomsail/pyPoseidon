"""
Schism model of pyposeidon. It controls the creation, execution & output  of a complete simulation based on SCHISM.

"""
# Copyright 2018 European Union
# This file is part of pyposeidon.
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by the European Commission - subsequent versions of the EUPL (the "Licence").
# Unless required by applicable law or agreed to in writing, software distributed under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and limitations under the Licence.

import os
import datetime
import numpy as np
import sys
import json
import pandas as pd
import glob
import xarray as xr
import geopandas as gp
import shapely
import jinja2
from searvey import ioc
import shutil
from scipy.spatial import Delaunay

# local modules
import pyposeidon
import pyposeidon.mesh as pmesh
import pyposeidon.meteo as pmeteo
import pyposeidon.dem as pdem
from pyposeidon.paths import DATA_PATH
from pyposeidon.utils.get_value import get_value
from pyposeidon.utils.converter import myconverter
from pyposeidon.utils.cpoint import closest_n_points
from pyposeidon.utils import data
from pyposeidon.utils.norm import normalize_column_names
from pyposeidon.utils.norm import normalize_varnames
from pyposeidon.utils.obs import get_obs_data
from pyposeidon.utils.post import export_xarray

# telemac (needs telemac stack on conda)
from telapy.api.t2d import Telemac2d
from telapy.api.t3d import Telemac3d
from telapy.api.art import Artemis
from telapy.api.wac import Tomawac
from execution.telemac_cas import TelemacCas
from data_manip.formats.selafin import Selafin
from data_manip.extraction.telemac_file import TelemacFile
from pretel.extract_contour import detecting_boundaries
from pretel.generate_atm import generate_atm
from pretel.extract_contour import sorting_boundaries
from utils.progressbar import ProgressBar
from pretel.manip_telfile import alter
from mpi4py import MPI

import numpy as np

import logging

from . import tools


logger = logging.getLogger(__name__)


TELEMAC_NAME = "telemac"


def divisors(n):
    result = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            result.add(i)
            result.add(n // i)
    return sorted(result)


def calculate_time_step_hourly_multiple(resolution, min_water_depth=0.1, courant_number=50):
    """
    Calculate the maximum allowable time step for a shallow water equation model
    based on the CFL condition, rounded to the closest divisor of an hour.

    :param resolution: Spatial resolution (Δx) in meters.
    :param min_water_depth: Minimum water depth (h_min) in meters. Default is 1 meter.
    :param courant_number: Courant number (C), typically less than 1 for explicit schemes. Can be higher for implicit schemes.
    :return: Maximum allowable time step (Δt) in seconds, rounded to the closest divisor of an hour.
    """
    resolution *= 111.389  # degrees to meters conversion factor
    raw_time_step = courant_number * resolution / (9.81 * min_water_depth) ** 0.5
    # Get all divisors of an hour (3600 seconds)
    hour_divisors = divisors(3600)
    # Find the closest divisor of an hour to the raw time step
    closest_divisor = min(hour_divisors, key=lambda x: abs(x - raw_time_step))
    return closest_divisor


# Helper functions
def remove(path):
    try:
        if os.path.isfile(path):
            os.remove(path)  # Remove a file
        elif os.path.isdir(path):
            if not os.listdir(path):  # Check if the directory is empty
                os.rmdir(path)
            else:
                shutil.rmtree(path)
    except OSError as e:
        print(f"Error: {e.strerror}")


def is_ccw(tris, meshx, meshy):
    x1, x2, x3 = meshx[tris].T
    y1, y2, y3 = meshy[tris].T
    return (y3 - y1) * (x2 - x1) > (y2 - y1) * (x3 - x1)


def flip(tris):
    return np.column_stack((tris[:, 2], tris[:, 1], tris[:, 0]))


def is_overlapping(tris, meshx):
    PIR = 180
    x1, x2, x3 = meshx[tris].T
    return np.logical_or(abs(x2 - x1) > PIR, abs(x3 - x1) > PIR)


def get_det_mask(tris, meshx, meshy):
    t12 = meshx[tris[:, 1]] - meshx[tris[:, 0]]
    t13 = meshx[tris[:, 2]] - meshx[tris[:, 0]]
    t22 = meshy[tris[:, 1]] - meshy[tris[:, 0]]
    t23 = meshy[tris[:, 2]] - meshy[tris[:, 0]]
    return t12 * t23 - t22 * t13 > 0


def contains_pole(x, y):
    return np.any(y == 90, axis=0)


def fix_glob_connectivity(x, y, ikle2, corrections):
    # Assuming slf.meshx, slf.meshy, and slf.ikle2 are NumPy arrays
    ikle2 = np.array(ikle2)
    # Ensure all triangles are CCW
    ccw_mask = is_ccw(ikle2, x, y)
    ikle2[~ccw_mask] = flip(ikle2[~ccw_mask])

    # triangles accross the dateline
    m_ = is_overlapping(ikle2, x)
    ikle2[m_] = flip(ikle2[m_])

    # special case : pole triangles
    pole_mask = contains_pole(x[ikle2].T, y[ikle2].T)

    # manual additional corrections
    if corrections is not None:
        for rev in corrections["reverse"]:
            ikle2[rev : rev + 1] = flip(ikle2[rev : rev + 1])
        for rem in corrections["remove"]:
            pole_mask[rem] = True

    # Check for negative determinant
    detmask = ~get_det_mask(ikle2, x, y)
    logger.debug(f"reversed {detmask.sum()} triangles")
    logger.debug(f"pole triangles: {np.where(pole_mask)[0]}")
    logger.debug("[TEMPORARY FIX]: REMOVE THE POLE TRIANGLES")
    return ikle2[~pole_mask, :]


def write_netcdf(ds, outpath):
    fileOut = os.path.splitext(outpath)[0] + ".nc"
    ds.to_netcdf(fileOut)


def write_meteo(outpath, geo, ds, gtype="grid", ttype="time", input360=False):
    lon = ds.longitude.values
    if input360:
        lon[lon < -180] += 360
    lat = ds.latitude.values
    #
    vars = list(ds.keys())
    var_used = []

    tmpFile = os.path.splitext(outpath)[0] + "_tmp.slf"

    remove(tmpFile)

    atm = TelemacFile(tmpFile, access="w")

    if gtype == "grid":
        nx1d = len(lon)
        ny1d = len(lat)
        x = np.tile(lon, ny1d).reshape(ny1d, nx1d).T.ravel()
        y = np.tile(lat, nx1d)
        nelem2 = 2 * (nx1d - 1) * (ny1d - 1)
        # set geometry
        ikle3 = np.zeros((nelem2, 3), dtype=np.int32)
        ielem = 0
        for i in range(1, nx1d):
            for j in range(1, ny1d):
                ipoin = (i - 1) * ny1d + j - 1
                # ~~> first triangle
                ikle3[ielem][0] = ipoin
                ikle3[ielem][1] = ipoin + ny1d
                ikle3[ielem][2] = ipoin + 1
                ielem += 1
                # ~~> second triangle
                ikle3[ielem][0] = ipoin + ny1d
                ikle3[ielem][1] = ipoin + ny1d + 1
                ikle3[ielem][2] = ipoin + 1
                ielem += 1
    else:
        x = lon
        y = lat
        tri = Delaunay(np.column_stack((lon, lat)))
        ikle3 = tri.simplices

    atm.add_mesh(x, y, ikle3)

    if ttype == "time":
        t0 = pd.Timestamp(ds.time.values[0])
        seconds = (pd.DatetimeIndex(ds.time.values) - t0).total_seconds()
    elif ttype == "step":
        t0 = pd.Timestamp(ds.time.values)
        seconds = ds.step.values / 1e9
    datetime = [t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second, 0]
    atm.add_header("pyposeidon generated mesh", date=datetime)

    # Meta data and variable names
    if "u10" in vars:
        atm.add_variable("WIND VELOCITY U", "M/S")
        var_used.append("u10")
    if "v10" in vars:
        atm.add_variable("WIND VELOCITY V", "M/S")
        var_used.append("v10")
    if "msl" in vars:
        atm.add_variable("SURFACE PRESSURE", "UI")
        var_used.append("msl")
    if "tmp " in vars:
        atm.add_variable("AIR TEMPERATURE ", "DEGREES")
        var_used.append("tmp")
    if not atm.varnames:
        raise ValueError("There are no meteorological variables to convert!")

    pbar = ProgressBar(len(seconds))
    atm._times = np.array(seconds, dtype=np.float64)
    atm._ntimestep = len(seconds)
    atm._values = np.zeros((len(seconds), atm._nvar, atm._npoin3))
    for itime, t_ in enumerate(seconds):
        pbar.update(itime)
        for var_ds, var_tel in zip(var_used, atm.varnames):
            data = np.ravel(np.transpose(ds.isel(time=itime)[var_ds].values))
            atm.add_data_value(var_tel, itime, data)
    atm.write()
    atm.close()

    # TEMP FIX NAMES CONVENTION for v8p5
    tmpFile2 = os.path.splitext(outpath)[0] + "_tmp2.slf"
    tmpFile3 = os.path.splitext(outpath)[0] + "_tmp3.slf"
    # interpolate on geo mesh
    generate_atm(geo, tmpFile, tmpFile2, None)
    remove(tmpFile3)
    alter(tmpFile2, tmpFile3, rename_var="WIND VELOCITY U=WINDX")
    remove(outpath)
    alter(tmpFile3, outpath,  rename_var="WIND VELOCITY V=WINDY")
    remove(tmpFile)
    remove(tmpFile2)
    remove(tmpFile3)


#
def get_boundary_settings(boundary_type, glo_node, bnd_node):
    settings = {
        "lihbor": 5 if boundary_type == "open" else 2,
        "liubor": 6 if boundary_type == "open" else 2,
        "livbor": 6 if boundary_type == "open" else 2,
        "hbor": 0.0,
        "ubor": 0.0,
        "vbor": 0.0,
        "aubor": 0.0,
        "litbor": 5 if boundary_type == "open" else 2,
        "tbor": 0.0,
        "atbor": 0.0,
        "btbor": 0.0,
        "nbor": glo_node + 1,
        "k": bnd_node + 1,
    }
    return settings


def format_value(value, width, precision=3, is_float=False):
    if is_float:
        return f"{value:{5}.{precision}f}"
    else:
        return f"{value:{2}}"


def detect_boundary_points_optimized(connectivity_table, npoints):
    # Building adjacency list from connectivity table
    adjacency_list = {i: [] for i in range(npoints)}
    for row in connectivity_table:
        for i in range(3):
            adjacency_list[row[i]].extend(row[np.arange(3) != i])

    buf = []
    connectivity_bnd_elt = []

    pbar = ProgressBar(npoints)
    for i in range(npoints):
        pbar.update(i)
        # Directly accessing the connections for each point
        connections = adjacency_list[i]

        uniq, count = np.unique(connections, return_counts=True)
        temp = uniq[count == 1].tolist()
        buf.extend(temp)

        if temp:
            if i in temp:
                temp.remove(i)
            temp.append(i)
            connectivity_bnd_elt.append(temp)

    pbar.finish()
    bnd_points = np.unique(buf)

    return np.array(connectivity_bnd_elt), bnd_points


def write_cli(inTel, ds, outCli=None, tel_module="telemac2d", global_=True):
    """
    (This function is a modification of the existing extract_contour() function
    in scripts/python3/pretel/extract_contour.py of the TELEMAC scripts)

    Generic function for extraction of contour from a mesh (with our without
    boundary file)

    @param inTel (str) Path to the mesh file
    @param ds (xr.Dataset) xarray Dataset of the mesh file (used to extract the boundary types)
    @param outCli (str) Path to the output contour file

    @returns (list) List of polygons
    """
    # TELEMAC script PART
    domains = []
    tel = TelemacFile(inTel)

    connectivity_bnd_elt, bnd_points = detect_boundary_points_optimized(tel.ikle2, tel.npoin2)

    boundaries_ordered, bnd_idx_left, bnd_elt_left = sorting_boundaries(
        tel, bnd_points, connectivity_bnd_elt, global_=global_
    )
    domains.append(boundaries_ordered)

    i = 1
    while bnd_elt_left.size != 0:
        i += 1

        boundaries_ordered, bnd_idx_left, bnd_elt_left = sorting_boundaries(
            tel, bnd_idx_left, bnd_elt_left, global_=global_
        )
        domains.append(boundaries_ordered)

    # custom pyposeidon part
    if outCli is None:
        outCli = os.path.splitext(inTel)[0] + ".cli"
    node_to_type = dict(zip(ds.node.values, ds.type.values))  # mapping from node to type
    domains_bnd = []
    lines = []
    bnd_node = 0
    for domain in domains:
        poly_bnd = []
        for bnd in domain:
            coord_bnd = []
            for i, glo_node in enumerate(bnd[:-1]):  # not taking the last node (not repeating)
                x, y = tel.meshx[glo_node], tel.meshy[glo_node]
                coord_bnd.append((x, y))
                # Determine boundary type for the current node
                boundary_type = node_to_type.get(glo_node, "Unknown")
                if boundary_type == "open":
                    # Get previous and next node indices in a circular manner
                    prev_node = bnd[i - 1] if i > 0 else bnd[-2]  # -2 to skip the repeated last node
                    next_node = bnd[i + 1] if i < len(bnd) - 2 else bnd[0]  # Wrapping around to the first node
                    # Get boundary types for previous and next nodes
                    prev_boundary_type = node_to_type.get(prev_node, "Unknown")
                    next_boundary_type = node_to_type.get(next_node, "Unknown")
                    # If both adjacent nodes are not 'open', then bnd is closed
                    if prev_boundary_type != "open" and next_boundary_type != "open":
                        boundary_type = "Unknown"
                boundary_settings = get_boundary_settings(boundary_type, glo_node, bnd_node)

                keys_order = [
                    "lihbor",
                    "liubor",
                    "livbor",
                    "hbor",
                    "ubor",
                    "vbor",
                    "aubor",
                    "litbor",
                    "tbor",
                    "atbor",
                    "btbor",
                    "nbor",
                    "k",
                ]
                if tel_module == "telemac2d":
                    line = " ".join(str(boundary_settings[key]) for key in keys_order)
                    lines.append(line)
                bnd_node += 1
            poly_bnd.append((coord_bnd, bnd))
        domains_bnd.append(poly_bnd)

    # Writing to file
    with open(outCli, "w") as f:
        for line in lines:
            formatted_line = " ".join(
                format_value(value, 3, is_float=isinstance(value, float)) for value in line.split()
            )
            f.write(f"{formatted_line}\n")

    return domains_bnd


def write_cas(outpath, tel_module, params):
    # Load the Jinja2 template
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.join(DATA_PATH)))

    # Define the custom filter for tidal bounds
    def repeat_value(value, count):
        return ";".join([str(value)] * count)

    env.filters["repeat_value"] = repeat_value
    template = env.get_template(tel_module + ".cas")
    ouput = template.render(params)
    with open(os.path.join(outpath, tel_module + ".cas"), "w") as f:
        f.write(ouput)


class Telemac:
    def __init__(self, **kwargs):
        """
        Create a TELEMAC model:
        by default : TELEMAC2D is used.

        Args:
            rfolder str: the TELEMAC module to use. Can be one of the following: telemac3d, telemac2d, artemis, tomawac


        """
        tel_module = kwargs.get("module", "telemac2d")
        if tel_module not in ["telemac3d", "telemac2d", "artemis", "tomawac"]:
            raise ValueError("module must be one of the following: telemac3d, telemac2d, artemis, tomawac")

        rfolder = kwargs.get("rfolder", None)
        if rfolder:
            self.read_folder(**kwargs)

        self.geometry = kwargs.get("geometry", None)

        if self.geometry:
            if isinstance(self.geometry, dict):
                self.lon_min = self.geometry["lon_min"]
                self.lon_max = self.geometry["lon_max"]
                self.lat_min = self.geometry["lat_min"]
                self.lat_max = self.geometry["lat_max"]
            elif self.geometry == "global":
                logger.warning("geometry is 'global'")
            elif isinstance(self.geometry, str):
                try:
                    geo = gp.GeoDataFrame.from_file(self.geometry)
                except:
                    logger.error("geometry argument not a valid geopandas file")
                    sys.exit(1)

                (
                    self.lon_min,
                    self.lat_min,
                    self.lon_max,
                    self.lat_max,
                ) = geo.total_bounds
            else:
                logger.warning("no geometry given")

        # coastlines
        coastlines = kwargs.get("coastlines", None)

        if coastlines is not None:
            try:
                coast = gp.GeoDataFrame.from_file(coastlines)
            except:
                coast = gp.GeoDataFrame(coastlines)

            self.coastlines = coast

        if not hasattr(self, "start_date"):
            start_date = kwargs.get("start_date", None)
            self.start_date = pd.to_datetime(start_date)

        if not hasattr(self, "end_date"):
            if "time_frame" in kwargs:
                time_frame = kwargs.get("time_frame", None)
                self.end_date = self.start_date + pd.to_timedelta(time_frame)
            elif "end_date" in kwargs:
                end_date = kwargs.get("end_date", None)
                self.end_date = pd.to_datetime(end_date)
                self.time_frame = self.end_date - self.start_date

        if not hasattr(self, "rdate"):
            self.rdate = get_value(self, kwargs, "rdate", self.start_date)

        if not hasattr(self, "end_date"):
            # ---------------------------------------------------------------------
            logger.warning("model not set properly, No end_date\n")
            # ---------------------------------------------------------------------

        self.tstep = kwargs.get("tstep", None)

        self.tag = tel_module
        self.tide = kwargs.get("tide", False)
        tpxo_ = kwargs.get("tpxo_source", None)
        if tpxo_ is not None:
            self.tide = True

        self.atm = kwargs.get("atm", True)
        self.monitor = kwargs.get("monitor", False)

        self.solver_name = TELEMAC_NAME
        # restart
        restart = get_value(self, kwargs, "hotstart", None)
        if restart:
            self.restart = pd.to_datetime(restart)
        else:
            self.restart = None

        # specific to meteo grib files
        self.gtype = get_value(self, kwargs, "meteo_gtype", "grid")
        self.ttype = get_value(self, kwargs, "meteo_ttype", "time")
        self.ncsize = get_value(self, kwargs, "ncsize", 1)
        # convert -180/180 to 0-360
        self.input360 = get_value(self, kwargs, "meteo_input360", False)

        for attr, value in kwargs.items():
            if not hasattr(self, attr):
                setattr(self, attr, value)

        setattr(self, "misc", {})

    # ============================================================================================
    # CONFIG
    # ============================================================================================
    def config(self, config_file=None, output=False, **kwargs):
        dic = get_value(self, kwargs, "parameters", None)
        #        param_file = get_value(self,kwargs,'config_file',None)

        if config_file:
            # ---------------------------------------------------------------------
            logger.info("reading parameter file {}\n".format(config_file))
            # ---------------------------------------------------------------------
        else:
            # ---------------------------------------------------------------------
            logger.info("using default " + self.tag + " json config file ...\n")
            # ---------------------------------------------------------------------

            config_file = os.path.join(DATA_PATH, self.tag + ".json")

        # Load the parameters from the JSON file
        with open(config_file, "r") as json_file:
            config = json.load(json_file)
            params = config["params"]

        # update key values
        if "telemac" in self.tag:
            params["datestart"] = self.start_date.strftime("%Y;%m;%d")
            params["timestart"] = self.start_date.strftime("%H;%M;%S")
        elif "tomawac" in self.tag:
            params["datestart"] = self.start_date.strftime("%Y%m%d%H%M")

        if hasattr(self, "time_frame"):
            duration = pd.to_timedelta(self.time_frame).total_seconds()
        else:
            self.time_frame = self.end_date - self.start_date
            duration = self.time_frame.total_seconds()

        params["duration"] = duration
        # export grid data every hour
        res_min = get_value(self, kwargs, "resolution_min", 0.5)

        if self.tstep:
            tstep = self.tstep
        else:
            tstep = calculate_time_step_hourly_multiple(res_min)
        params["tstep"] = tstep

        params["nb_tsteps"] = int(duration / tstep)
        params["tstep_graph"] = int(3600 / tstep)
        params["tstep_list"] = int(3600 / tstep)
        params["ncsize"] = self.ncsize

        # tide
        if self.tide:
            params["tide"] = True
            params["initial_conditions"] = "TPXO SATELLITE ALTIMETRY"

        # hotstart
        if self.restart is not None:
            hotout = int((self.restart - self.rdate).total_seconds() / (params["tstep"] * params["tstep_graph"]))
            params["hotstart"] = True
            params["restart_tstep"] = hotout

        # update
        if dic:
            for key in dic.keys():
                params[key] = dic[key]
        if "params" in kwargs.keys():
            for key in kwargs["params"].keys():
                params[key] = kwargs["params"][key]

        self.params = params

        if output:
            # save params
            # ---------------------------------------------------------------------
            logger.info("output " + self.tag + " CAS file ...\n")
            # ---------------------------------------------------------------------
            path = get_value(self, kwargs, "rpath", "./telemac/")
            write_cas(path, self.tag, params)

    # ============================================================================================
    # METEO
    # ============================================================================================
    def force(self, **kwargs):
        meteo_source = get_value(self, kwargs, "meteo_source", None)

        kwargs.update({"meteo_source": meteo_source})

        flag = get_value(self, kwargs, "update", ["all"])

        z = {**self.__dict__, **kwargs}  # merge self and possible kwargs

        # check if files exist
        if flag:
            if ("meteo" in flag) | ("all" in flag):
                self.meteo = pmeteo.Meteo(**z)
            else:
                logger.info("skipping meteo ..\n")
        else:
            self.meteo = pmeteo.Meteo(**z)

        if hasattr(self, "meteDataseto"):
            # add 1 hour for Schism issue with end time
            ap = self.meteo.Dataset.isel(time=-1)
            ap["time"] = ap.time.values + pd.to_timedelta("1H")

            self.meteo.Dataset = xr.concat([self.meteo.Dataset, ap], dim="time")

    def to_force(self, geo, outpath):
        # # WRITE METEO FILE
        logger.info("saving meteo file.. ")
        out_file = os.path.join(outpath, "input_wind.slf")
        write_meteo(out_file, geo, self.meteo.Dataset, gtype=self.gtype, ttype=self.ttype, input360=self.input360)

    # ============================================================================================
    # TPXO
    # ============================================================================================
    def tpxo(self, tpxo_suffix="tpxo9_atlas_30_v4", **kwargs):
        tpxo_source = get_value(self, kwargs, "tpxo_source", None)
        path = get_value(self, kwargs, "rpath", "./telemac")

        lat_min = np.round(self.lat_min - 1 if self.lat_min > -89 else self.lat_min, 2)
        lat_max = np.round(self.lat_max + 1 if self.lat_max < 89 else self.lat_max, 2)
        lon_min = np.round(self.lon_min - 1, 2)
        lon_max = np.round(self.lon_max + 1, 2)

        if lon_min < 0 and lon_max < 0:
            lon_min += 360
            lon_max += 360

        if tpxo_source is None:
            # ---------------------------------------------------------------------
            logger.error("model not set properly: define tpxo_source\n")
            raise ValueError("model not set properly: define tpxo_source")
            # ---------------------------------------------------------------------
        os.makedirs(os.path.join(path, "TPXO"), exist_ok=True)

        logger.info("extracting tides..\n")
        #
        lines = []
        lines.append(f"{path}/TPXO/input_sources !")
        lines.append(f"{path}/TPXO/Model_Local        ! Local area Model name")
        lines.append(f"{lat_min} {lat_max}     ! Lat limits (degrees N)")
        lines.append(f"{lon_min} {lon_max}     ! Lon limits (degrees E, range -180 +360)")
        with open(f"{path}/TPXO/setup.local", "w") as file:
            file.write("\n".join(lines))

        # input sources
        lines = []
        lines.append(f"{tpxo_source}/h_*_{tpxo_suffix}")
        lines.append(f"{tpxo_source}/u_*_{tpxo_suffix}")
        lines.append(f"{tpxo_source}/grid_{tpxo_suffix}")
        with open(f"{path}/TPXO/input_sources", "w") as file:
            file.write("\n".join(lines))

        # output files
        lines = []
        lines.append(f"{path}/TPXO/h_LOCAL")
        lines.append(f"{path}/TPXO/uv_LOCAL")
        lines.append(f"{path}/TPXO/grid_LOCAL")
        with open(f"{path}/TPXO/Model_Local", "w") as file:
            file.write("\n".join(lines))

        os.system(f"cp {DATA_PATH}/extract_local_model {path}/TPXO/extract_local_model")
        os.system(f"{path}/TPXO/extract_local_model<{path}/TPXO/setup.local")

    # ============================================================================================
    # DEM
    # ============================================================================================
    def bath(self, **kwargs):
        #       z = self.__dict__.copy()

        if self.mesh.Dataset is not None:
            kwargs["grid_x"] = self.mesh.Dataset.SCHISM_hgrid_node_x.values
            kwargs["grid_y"] = self.mesh.Dataset.SCHISM_hgrid_node_y.values

        dpath = get_value(self, kwargs, "dem_source", None)

        kwargs.update({"dem_source": dpath})

        flag = get_value(self, kwargs, "update", ["all"])
        # check if files exist
        if flag:
            if ("dem" in flag) | ("all" in flag):
                kwargs.update(
                    {
                        "lon_min": self.lon_min,
                        "lat_min": self.lat_min,
                        "lon_max": self.lon_max,
                        "lat_max": self.lat_max,
                    }
                )
                self.dem.Dataset = pdem.dem_on_mesh(self.dem.Dataset, **kwargs)

                if kwargs.get("adjust_dem", True):
                    coastline = kwargs.get("coastlines", None)
                    if coastline is None:
                        logger.warning("coastlines not present, aborting adjusting dem\n")
                    elif "adjusted" in self.dem.Dataset.data_vars.keys():
                        logger.info("Dem already adjusted\n")
                    elif "fval" in self.dem.Dataset.data_vars.keys():
                        logger.info("Dem already adjusted\n")
                    else:
                        self.dem.adjust(coastline, **kwargs)
                try:
                    try:
                        bat = -self.dem.Dataset.fval.values.astype(float)  # minus for the hydro run
                        if np.isinf(bat).sum() != 0:
                            raise Exception("Bathymetry contains Infs")
                        if np.isnan(bat).sum() != 0:
                            raise Exception("Bathymetry contains NaNs")
                            logger.warning("Bathymetric values fval contain NaNs, using ival values ..\n")

                    except:
                        bat = -self.dem.Dataset.ival.values.astype(float)  # minus for the hydro run

                    self.mesh.Dataset.depth.loc[: bat.size] = bat

                    logger.info("updating bathymetry ..\n")

                except AttributeError as e:
                    logger.info("Keeping bathymetry in hgrid.gr3 due to {}\n".format(e))

            else:
                logger.info("dem from mesh file\n")

    def to_slf(self, outpath, global_=True, friction_type="chezy", **kwargs):
        corrections = get_value(self, kwargs, "mesh_corrections", {"reverse": [], "remove": []})
        now = pd.Timestamp.now()
        _date = [now.year, now.month, now.day, now.hour, now.minute, 0]

        X = self.mesh.Dataset.SCHISM_hgrid_node_x.data
        Y = self.mesh.Dataset.SCHISM_hgrid_node_y.data
        # depth
        Z = self.mesh.Dataset.depth.data
        nan_mask = np.isnan(Z)
        inf_mask = np.isinf(Z)
        if np.any(nan_mask) or np.any(inf_mask):
            Z[nan_mask] = 0
            Z[inf_mask] = 0
            logger.info("Z contains . Cleaning needed.")
        # connectivity
        IKLE2 = self.mesh.Dataset.SCHISM_hgrid_face_nodes.data
        remove(outpath)

        if global_:
            # adjust triangles orientation on the dateline
            # and also suppress the pole triangles (for now)
            IKLE2 = fix_glob_connectivity(X, Y, IKLE2, corrections)

        res = TelemacFile(outpath, access="w")
        res.add_header("pyposeidon generated mesh", date=_date)
        res.add_mesh(X, Y, IKLE2)
        # add BOTTOM variable
        res.add_variable("BOTTOM", "M")
        res._values[0, 0, :] = Z
        # bottom friction only in the case of TELEMAC2D
        if self.tag == "telemac2d":
            chezy = get_value(self, kwargs, "chezy", None)
            if chezy:
                C = np.ones(len(Z))
            else:
                manning = get_value(self, kwargs, "manning", 0.027)
                if friction_type == "chezy":
                    C = (abs(Z) ** (1 / 6)) / manning
                else:
                    print("only Chezy implemented so far! ")
                    sys.exit()
            res.add_variable("BOTTOM FRICTION", "")
            res._values[0, 1, :] = C
            logger.info("Manning file created..\n")
        #
        res.write()
        res.close()
        return res

    # ============================================================================================
    # EXECUTION
    # ============================================================================================
    def create(self, **kwargs):
        if not kwargs:
            kwargs = self.__dict__.copy()

        # Set background dem as scale for mesh generation
        dpath = get_value(self, kwargs, "dem_source", None)

        if dpath:
            self.dem = pdem.Dem(**kwargs)
            # kwargs.update({"dem_source": self.dem.Dataset})
        else:
            logger.info("no dem available\n")

        # Mesh
        self.mesh = pmesh.set(type="tri2d", **kwargs)

        # set lat/lon from file
        if self.mesh.Dataset is not None:
            kwargs.update({"lon_min": self.mesh.Dataset.SCHISM_hgrid_node_x.values.min()})
            kwargs.update({"lon_max": self.mesh.Dataset.SCHISM_hgrid_node_x.values.max()})
            kwargs.update({"lat_min": self.mesh.Dataset.SCHISM_hgrid_node_y.values.min()})
            kwargs.update({"lat_max": self.mesh.Dataset.SCHISM_hgrid_node_y.values.max()})

            self.lon_min = self.mesh.Dataset.SCHISM_hgrid_node_x.values.min()
            self.lon_max = self.mesh.Dataset.SCHISM_hgrid_node_x.values.max()
            self.lat_min = self.mesh.Dataset.SCHISM_hgrid_node_y.values.min()
            self.lat_max = self.mesh.Dataset.SCHISM_hgrid_node_y.values.max()

        # get bathymetry
        if self.mesh.Dataset is not None:
            self.bath(**kwargs)

        # get meteo
        if self.atm:
            self.force(**kwargs)

        # get tide
        if self.tide:
            self.tpxo(**kwargs)

        self.config(**kwargs)

    def output(self, **kwargs):
        path = get_value(self, kwargs, "rpath", "./telemac/")
        flag = get_value(self, kwargs, "update", ["all"])
        split_by = get_value(self, kwargs, "meteo_split_by", None)

        if not os.path.exists(path):
            os.makedirs(path)

        # Mesh related files
        if self.mesh.Dataset is not None:
            # WRITE GEO FILE
            try:
                bat = self.dem.Dataset.fval.values.astype(float)  # minus for the hydro run
                if np.isnan(bat).sum() != 0:
                    raise Exception("Bathymetry contains NaNs")
                    logger.warning("Bathymetric values fval contain NaNs, using ival values ..\n")

            except:
                bat = self.dem.Dataset.ival.values.astype(float)  # minus for the hydro run
            self.mesh.Dataset.depth.loc[: bat.size] = bat

            logger.info("saving geometry file.. ")
            geo = os.path.join(path, "geo.slf")
            slf = self.to_slf(geo, global_=True)
            write_netcdf(self.mesh.Dataset, geo)

            # # WRITE METEO FILE
            logger.info("saving meteo file.. ")
            meteo = os.path.join(path, "input_wind.slf")
            self.atm = write_meteo(
                meteo, geo, self.meteo.Dataset, gtype=self.gtype, ttype=self.ttype, input360=self.input360
            )

            # WRITE BOUNDARY FILE
            logger.info("saving boundary file.. ")
            domain = write_cli(geo, self.mesh.Dataset)
            self.params["N_bc"] = len(domain[0])

            # WRITE CAS FILE
            logger.info("saving CAS file.. ")
            write_cas(path, self.tag, self.params)

        # ---------------------------------------------------------------------
        logger.info("output done\n")
        # ---------------------------------------------------------------------

    def run(self, **kwargs):
        calc_dir = get_value(self, kwargs, "rpath", "./telemac/")
        cwd = os.getcwd()

        # ---------------------------------------------------------------------
        logger.info("executing model\n")
        # ---------------------------------------------------------------------
        comm = MPI.COMM_WORLD

        cas_file = self.tag + ".cas"
        os.chdir(calc_dir)
        if not tools.is_mpirun_installed():
            logger.warning("mpirun is not installed, ending.. \n")
            return

        # Creation of the instance Telemac2d
        study = Telemac2d(cas_file, user_fortran=None, comm=comm, stdout=0, recompile=True)

        # Testing construction of variable list
        _ = study.variables

        study.set_case()
        # Initialization
        study.init_state_default()
        # Run all time steps
        ntimesteps = study.get("MODEL.NTIMESTEPS")
        pbar = ProgressBar(ntimesteps)
        for it in range(ntimesteps):
            study.run_one_time_step()
            pbar.update(it)

        # Ending the run
        study.finalize()
        pbar.finish()
        #
        os.chdir(cwd)

    def save(self, **kwargs):
        path = get_value(self, kwargs, "rpath", "./telemac/")

        lista = [key for key, value in self.__dict__.items() if key not in ["meteo", "dem", "mesh"]]
        dic = {k: self.__dict__.get(k, None) for k in lista}

        mesh = self.__dict__.get("mesh", None)
        if isinstance(mesh, str):
            dic.update({"mesh": mesh})
        else:
            dic.update({"mesh": mesh.__class__.__name__})

        dem = self.__dict__.get("dem", None)
        if isinstance(dem, str):
            dic.update({"dem": dem})
        elif isinstance(dem, pdem.Dem):
            dic.update({"dem_attrs": dem.Dataset.elevation.attrs})

        meteo = self.__dict__.get("meteo", None)
        if isinstance(meteo, str):
            dic.update({"meteo": meteo})
        elif isinstance(meteo, pmeteo.Meteo):
            try:
                dic.update({"meteo": [meteo.Dataset.attrs]})
            except:
                dic.update({"meteo": [x.attrs for x in meteo.Dataset]})

        coastlines = self.__dict__.get("coastlines", None)
        if coastlines is not None:
            # Save the path to the serialized coastlines - #130
            coastlines_database = os.path.join(path, "coastlines.json")
            coastlines.to_file(coastlines_database)
            dic.update({"coastlines": coastlines_database})
        else:
            dic.update({"coastlines": None})

        dic["version"] = pyposeidon.__version__

        for attr, value in dic.items():
            if isinstance(value, datetime.datetime):
                dic[attr] = value.isoformat()
            if isinstance(value, pd.Timedelta):
                dic[attr] = value.isoformat()
            if isinstance(value, pd.DataFrame):
                dic[attr] = value.to_dict()
            if isinstance(value, gp.GeoDataFrame):
                dic[attr] = value.to_json()

        filename = os.path.join(path, f"{self.tag}_model.json")
        json.dump(dic, open(filename, "w"), indent=4, default=myconverter)

    def execute(self, **kwargs):
        self.create(**kwargs)
        self.output(**kwargs)
        if self.monitor:
            self.set_obs()
        self.save(**kwargs)
        self.run(**kwargs)

    def read_folder(self, rfolder, **kwargs):
        self.rpath = rfolder
        geo = glob.glob(os.path.join(rfolder, "/geo.slf"))
        cli = glob.glob(os.path.join(rfolder, "/geo.cli"))
        mfiles = glob.glob(os.path.join(rfolder, "/input_wind.slf"))

        mykeys = ["start_year", "start_month", "start_day"]
        sdate = [self.params["opt"][x] for x in mykeys]  # extract date info from param.nml
        sd = "-".join(str(item) for item in sdate)  # join in str
        sd = pd.to_datetime(sd)  # convert to datestamp

        sh = [self.params["opt"][x] for x in ["start_hour", "utc_start"]]  # get time attrs
        sd = sd + pd.to_timedelta("{}H".format(sh[0] + sh[1]))  # complete start_date

        ed = sd + pd.to_timedelta("{}D".format(self.params["core"]["rnday"]))  # compute end date based on rnday

        self.start_date = sd  # set attrs
        self.end_date = ed

        load_mesh = get_value(self, kwargs, "load_mesh", False)

        if load_mesh:
            try:
                self.mesh = pmesh.set(type="tri2d", mesh_file=hfile)
            except:
                logger.warning("loading mesh failed")
                pass
        else:
            logger.warning("no mesh loaded")

        load_meteo = get_value(self, kwargs, "load_meteo", False)

        # meteo
        if load_meteo is True:
            try:
                pm = []
                for key, val in mfiles.items():
                    msource = xr.open_mfdataset(val)
                    pm.append(msource)
                if len(pm) == 1:
                    pm = pm[0]
                self.meteo = pmeteo.Meteo(meteo_source=pm)

            except:
                pm = []
                for key, val in mfiles.items():
                    ma = []

                    for ifile in mfiles:
                        g = xr.open_dataset(ifile)
                        ts = "-".join(g.time.attrs["base_date"].astype(str)[:3])
                        time_r = pd.to_datetime(ts)
                        try:
                            times = time_r + pd.to_timedelta(g.time.values, unit="D").round("H")
                            g = g.assign_coords({"time": times})
                            ma.append(g)
                        except:
                            logger.warning("loading meteo failed")
                            break
                    if ma:
                        msource = xr.merge(ma)
                        pm.append(msource)

                if len(pm) == 1:
                    pm = pm[0]

                self.meteo = pmeteo.Meteo(meteo_source=pm)

        else:
            logger.warning("no meteo loaded")

    def hotstart(self, t=None, **kwargs):
        filename2d = get_value(self, kwargs, "filename2d", "out_2D.zarr")
        ppath = get_value(self, kwargs, "ppath", self.rpath)
        out2d = os.path.join(ppath, "outputs", filename2d)
        ds = xr.open_dataset(out2d)
        t_ = ds.time.values
        it = np.where(t_ == t)[0]
        xdat = ds.isel(time=it)
        hfile = f"hotstart_{t.strftime('%Y%m%d.%H')}.nc"
        logger.info("saving hotstart file\n")
        xdat.to_netcdf(os.path.join(ppath, f"outputs/{hfile}"))

    ## Any variable
    def combine(self, out, g2l, name):
        keys = g2l.index.get_level_values(0).unique()
        r = []
        for i in range(len(out)):
            v = out[i].to_pandas()
            v.index += 1
            mask = g2l.loc[keys[i], "local"]
            vm = v.loc[mask]
            vm.index = g2l.loc[keys[i], "global_n"].values
            r.append(vm)
        r = pd.concat(r).sort_index()
        r.index -= 1
        r.index.name = name
        return r

    def combine_(self, var, out, g2l, name):
        if len(out[0][var].shape) == 3:
            dd = []
            for i in range(out[0][var].shape[2]):
                o = []
                for j in range(len(out)):
                    o.append(out[j][var].loc[{out[j][var].dims[2]: i}])
                r = self.combine(o, g2l, name)
                dd.append(xr.DataArray(r.values, dims=out[0][var].dims[:2], name=var))

            tr = xr.concat(dd, dim=out[0][var].dims[2])
            a, b, c = out[0][var].dims
            tr = tr.transpose(a, b, c)
            return tr

        elif len(out[0][var].shape) < 3:
            o = []
            for j in range(len(out)):
                o.append(out[j][var])
            r = self.combine(o, g2l, name)
            return xr.DataArray(r.values, dims=list(out[0][var].dims), name=var)

    def xcombine(self, tfs):
        # Create dataset
        side = []
        node = []
        el = []
        single = []

        for key in tfs[0].variables:
            if "nSCHISM_hgrid_face" in tfs[0][key].dims:
                r = self.combine_(key, tfs, self.misc["melems"], "nSCHISM_hgrid_face")
                el.append(r)
            elif "nSCHISM_hgrid_node" in tfs[0][key].dims:
                r = self.combine_(key, tfs, self.misc["mnodes"], "nSCHISM_hgrid_node")
                node.append(r)
            elif "nSCHISM_hgrid_edge" in tfs[0][key].dims:
                r = self.combine_(key, tfs, self.misc["msides"], "nSCHISM_hgrid_edge")
                side.append(r)
            elif len(tfs[0][key].dims) == 1:
                single.append(tfs[0][key])

        side = xr.merge(side)
        el = xr.merge(el)
        node = xr.merge(node)
        single = xr.merge(single)

        # merge
        return xr.merge([side, el, node, single])

    def tcombine(self, hfiles, sdate, times):
        xall = []
        for k in range(times.shape[0]):
            tfs = []
            for i in range(len(hfiles)):
                tfs.append(xr.open_dataset(hfiles[i]).isel(time=k))

            xall.append(self.xcombine(tfs))

        xdat = xr.concat(xall, dim="time")
        xdat = xdat.assign_coords(time=times)

        return xdat

    # https://stackoverflow.com/questions/41164630/pythonic-way-of-removing-reversed-duplicates-in-list
    @staticmethod
    def remove_reversed_duplicates(iterable):
        # Create a set for already seen elements
        seen = set()
        for item in iterable:
            # Lists are mutable so we need tuples for the set-operations.
            tup = tuple(item)
            if tup not in seen:
                # If the tuple is not in the set append it in REVERSED order.
                seen.add(tup[::-1])
                # If you also want to remove normal duplicates uncomment the next line
                # seen.add(tup)
                yield item

    def results(self, **kwargs):
        lat_coord_standard_name = "latitude"
        lon_coord_standard_name = "longitude"
        x_units = "degrees_east"
        y_units = "degrees_north"

        path = get_value(self, kwargs, "rpath", "./telemac/")
        filename = get_value(self, kwargs, "filename", "stations.zarr")
        filename2d = get_value(self, kwargs, "filename2d", "out_2D.zarr")
        chunk = get_value(self, kwargs, "chunk", None)

        logger.info("get combined 2D netcdf files \n")
        # check for new IO output
        res = os.path.join(path, "results_2D.slf")
        slf = Selafin(res)

        year, month, day, hour, minute, seconds, tz = slf.datetime
        hour = "{0:05.2f}".format(float(hour)).replace(".", ":")
        tz = "{0:+06.2f}".format(float(0)).replace(".", "")
        date = "{}-{}-{} {} {}".format(year, month, day, hour, tz)
        # set the start timestamp
        sdate = pd.to_datetime(date, utc=True, format="%Y-%m-%d %H:%M %z")
        # Normalize Time Data
        sdate = pd.to_datetime(date, utc=True, format="%Y-%m-%d %H:%M %z")
        times = pd.to_datetime(slf.tags["times"], unit="s", origin=sdate.tz_convert(None))
        if isinstance(self.mesh, str):
            if self.mesh == "tri2d" or self.mesh == "r2d":
                self.mesh = pmesh.set(type=self.mesh, mesh_file=self.mesh_file)
        # Initialize first the xr.Dataset
        xc = xr.Dataset(
            {
                "longitude": ("nodes", slf.meshx),
                "latitude": ("nodes", slf.meshy),
                "face_node_connectivity": (("elements", "face_nodes"), slf.ikle2),
                "node": ("bnodes", self.mesh.Dataset.node.values),
                "type": ("bnodes", self.mesh.Dataset.type.values),
                "id": ("bnodes", self.mesh.Dataset.id.values),
                "depth": ("nodes", self.mesh.Dataset.depth.values),
            },
            coords={
                "bnodes": np.arange(len(self.mesh.Dataset.node.values)),
                "time": times,
                "nodes": np.arange(len(slf.meshx)),
            },
        )

        # Normalize Variable Names and Add Model Results
        varnames_n = normalize_varnames(slf.varnames)
        cube = np.zeros((len(times), len(varnames_n), len(slf.meshx)))

        for it, t_ in enumerate(slf.tags["times"]):
            # Get model results for this time step
            cube[it, :, :] = slf.get_values(it)  # This should return a 2D array of shape (n_var, nodes)
        # reindex
        for idx, varname in enumerate(varnames_n):
            xc = xc.assign({varname: (("time", "nodes"), cube[:, idx, :])})

        # set Attrs
        xc.longitude.attrs = {
            "long_name": "node x-coordinate",
            "standard_name": lon_coord_standard_name,
            "units": x_units,
            "mesh": "TELEMAC_Selafin",
        }

        xc.latitude.attrs = {
            "long_name": "node y-coordinate",
            "standard_name": lat_coord_standard_name,
            "units": y_units,
            "mesh": "TELEMAC_Selafin",
        }

        if "depth" in xc.variables:
            xc.depth.attrs = {
                "long_name": "Bathymetry",
                "units": "meters",
                "positive": "down",
                "mesh": "TELEMAC_Selafin",
                "location": "node",
            }

        xc.face_node_connectivity.attrs = {
            "long_name": "Horizontal Element Table",
            "cf_role": "face_node_connectivity",
            "start_index": 0,
        }

        xc.time.attrs = {
            "long_name": "Time",
            "base_date": date,
            "standard_name": "time",
        }

        # Dataset Attrs
        xc.attrs = {
            "Conventions": "CF-1.0, UGRID-1.0",
            "title": "TELEMAC Model output",
            "source": "TELEMAC model output version vV8P4",
            "references": "https://gitlab.pam-retd.fr/otm/telemac-mascaret",
            "history": "created by pyposeidon",
            "comment": "TELEMAC Model output",
            "type": "TELEMAC Model output",
        }
        os.makedirs(os.path.join(path, "outputs"), exist_ok=True)
        out2d = os.path.join(path, "outputs", filename2d)
        remove(out2d)
        export_xarray(xc, out2d, chunk=chunk)

        if self.monitor:
            logger.info("export observations nc file\n")
            # Normalize Variable Names and Add Model Results
            varnames_n = normalize_varnames(slf.varnames)
            cube = np.zeros((len(times), len(varnames_n), len(self.stations_mesh_id["gindex"])))
            if isinstance(self.stations_mesh_id, dict):
                stations = pd.DataFrame(self.stations_mesh_id)
            elif isinstance(self.stations_mesh_id, pd.DataFrame):
                stations = self.stations_mesh_id
            for it, t_ in enumerate(slf.tags["times"]):
                # Get model results for this time step
                data = slf.get_values(it)  # This should return a 2D array of shape (n_var, nodes)
                for i, idx in enumerate(stations["gindex"]):
                    idx = int(idx)
                    for var in range(len(varnames_n)):
                        if var >= data.shape[0]:
                            raise IndexError(f"Variable index {var} out of bounds for data with shape {data.shape}")
                        if idx >= data.shape[1]:
                            raise IndexError(f"Index {idx} out of bounds for data with shape {data.shape}")
                        cube[it, var, i] = data[var, idx]

            ds = xr.Dataset(
                {
                    "elev": (("time", "variable", "id"), cube),  # replace with your actual data
                },
                coords={"time": times, "id": stations["provider_id"], "variable": varnames_n},
            )

            out_obs = os.path.join(path, "outputs", filename)
            remove(out_obs)
            export_xarray(ds, out_obs, chunk=chunk)

        logger.info("done with output netCDF files \n")

    def set_obs(self, **kwargs):
        path = get_value(self, kwargs, "rpath", "./telemac/")
        nspool_sta = get_value(self, kwargs, "nspool_sta", 1)
        tg_database = get_value(self, kwargs, "obs", None)

        if tg_database:
            logger.info("get stations from {}\n".format(tg_database))
            tg = gp.read_file(tg_database)
        else:
            logger.info("get stations using searvey\n")
            geometry = get_value(self, kwargs, "geometry", None)
            if geometry == "global":
                tg = ioc.get_ioc_stations()
            else:
                geo_box = shapely.geometry.box(self.lon_min, self.lat_min, self.lon_max, self.lat_max)
                tg = ioc.get_ioc_stations(region=geo_box)
            tg = tg.reset_index(drop=True)
            ### save in compatible to searvey format
            tg["country"] = tg.country.values.astype("str")  # fix an issue with searvey see #43 therein
            logger.info("save station DataFrame \n")
            tg_database = os.path.join(path, "stations.json")
            tg.to_file(tg_database)
        ##### normalize to be used inside pyposeidon
        tgn = normalize_column_names(tg.copy())

        ##### make sure lat/lon are floats
        tgn = tgn.astype({"latitude": float, "longitude": float, "location": str})

        ## FOR TIDE GAUGE MONITORING

        logger.info("set in-situ measurements locations \n")

        if not self.mesh.Dataset:
            logger.warning("no mesh available skipping \n")
            return

        gpoints = np.array(
            list(
                zip(
                    self.mesh.Dataset.SCHISM_hgrid_node_x.values,
                    self.mesh.Dataset.SCHISM_hgrid_node_y.values,
                )
            )
        )

        coords = np.array([tgn.longitude.values, tgn.latitude.values]).T
        cp = closest_n_points(coords, 1, gpoints).T[0]
        mesh_index = cp
        stations = gpoints[cp]

        if len(stations) == 0:
            logger.warning("no observations available\n")

        stations = pd.DataFrame(stations, columns=["SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y"])
        stations["z"] = 0
        stations.index += 1
        stations["gindex"] = mesh_index
        try:
            stations["provider_id"] = tgn.location.values
            stations["provider"] = "ioc"
            stations["longitude"] = tgn.longitude.values
            stations["latitude"] = tgn.latitude.values
        except:
            pass

        self.stations_mesh_id = stations

        logger.info("write out stations.in file \n")

        # output to file
        with open(os.path.join(path, "station.in"), "w") as f:
            f.write(f"1 {stations.shape[0]}\n")  # 1st line: number of periods and number of points
            f.write(
                f"{0} {int(self.params['duration'])} {self.params['tstep']}\n"
            )  # 2nd line: period 1: start time, end time and interval (in seconds)
            stations.loc[:, ["SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y", "gindex"]].to_csv(
                f, header=None, sep=" "
            )  # 3rd-10th line: output points; x coordinate, y coordinate, station number, and station name

    def get_station_sim_data(self, **kwargs):
        path = get_value(self, kwargs, "rpath", "./schism/")

        # locate the station files
        sfiles = glob.glob(os.path.join(path, "outputs/staout_*"))
        sfiles.sort()

        try:
            # get the station flags
            flags = pd.read_csv(os.path.join(path, "station.in"), header=None, nrows=1, delim_whitespace=True).T
        except FileNotFoundError:
            logger.error("no station.in file present")
            return

        flags.columns = ["flag"]
        flags["variable"] = [
            "elev",
            "air_pressure",
            "windx",
            "windy",
            "T",
            "S",
            "u",
            "v",
            "w",
        ]

        vals = flags[flags.values == 1]  # get the active ones

        dstamp = kwargs.get("dstamp", self.rdate)

        dfs = []
        for idx in vals.index:
            obs = np.loadtxt(sfiles[idx])
            df = pd.DataFrame(obs)
            df = df.set_index(0)
            df.index.name = "time"
            df.columns.name = vals.loc[idx, "variable"]
            df.index = pd.to_datetime(dstamp) + pd.to_timedelta(df.index, unit="S")
            pindex = pd.MultiIndex.from_product([df.T.columns, df.T.index])

            r = pd.DataFrame(df.values.flatten(), index=pindex, columns=[vals.loc[idx, "variable"]])
            r.index.names = ["time", "node"]

            r.index = r.index.set_levels(r.index.levels[1] - 1, level=1)

            dfs.append(r.to_xarray())

        self.station_sim_data = xr.combine_by_coords(dfs)

    def get_output_data(self, **kwargs):
        dic = self.__dict__

        dic.update(kwargs)

        self.data = data.get_output(**dic)

    def get_station_obs_data(self, **kwargs):
        self.station_obs_data = get_obs_data(self.obs, self.start_date, self.end_date)

    def open_thalassa(self, **kwargs):
        # open a Thalassa instance to visualize the output
        return
