from datashader import transfer_functions as tf
import datashader as ds
import xarray as xr
from fastapi import HTTPException
import morecantile

from matplotlib import cm
from functools import partial
from numpy import uint8
from PIL import Image

# From Morecantile, morecantile.tms.list()
WEB_CRS = {
    3857: "WebMercatorQuad",
    32631: "UTM31WGS84Quad",
    3978: "CanadianNAD83_LCC",
    5482: "LINZAntarticaMapTilegrid",
    4326: "WorldCRS84Quad",
    5041: "UPSAntarcticWGS84Quad",
    3035: "EuropeanETRS89_LAEAQuad",
    3395: "WorldMercatorWGS84Quad",
    2193: "NZTM2000",
}


class DataValidationError(KeyError):
    pass


class FieldValidator:
    def __init__(self, validators=(), default=None):
        self.validators = validators
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if not instance:
            return self
        return instance.__dict__.get(self.name, self.default)

    def __delete__(self, instance):
        del instance.__dict__[self.name]

    def __set__(self, instance, value):
        for validator in self.validators:
            if value is not self:
                validator(value)
                instance.__dict__[self.name] = value
            else:
                instance.__dict__[self.name] = self.default


def validate_crs(crs_epsg):
    if crs_epsg not in WEB_CRS.keys():
        raise DataValidationError(f"User input {crs_epsg} not supported")


def validate_color_mapping(value):
    if value.get("datashader_settings") is None:
        raise DataValidationError(
            f"The key 'datashader_settings' is missing from {value}"
        )


def query_builder(time, xleft, xright, ybottom, ytop, xlab, ylab):
    query = {}
    query.update({xlab: slice(xleft, xright), ylab: slice(ytop, ybottom)})
    if time:
        query["time"] = time
    return query


def get_bounds(TMS, zoom, x, y):

    bbx = TMS.xy_bounds(morecantile.Tile(int(x), int(y), int(zoom)))

    return bbx.left, bbx.right, bbx.bottom, bbx.top


def get_tiles(var, dataset, query) -> xr.DataArray:

    tile = dataset[var].sel(query)

    if 0 in tile.sizes.values():
        raise HTTPException(status_code=406, detail=f"Map outside dataset domain")

    return tile


class Base:
    def __init__(
        self, interpolation={}, aggregation={}, normalization={}, color_mapping={}
    ):
        self.interp_params = interpolation
        self.agg_params = aggregation
        self.norm_params = normalization
        self.cm_params = color_mapping

    def interpolation(self, arr):
        return arr

    def aggregation(self, arr):
        return arr

    def normalization(self, arr):
        return arr

    def color_mapping(self, arr):
        return arr


class DataShaderBase(Base):
    def __init__(self, aggregation={}, color_mapping={}):
        super().__init__(aggregation=aggregation, color_mapping=color_mapping)

    def aggregation(self, tile):
        self.cvs = ds.Canvas(plot_width=256, plot_height=256)
        agg = self.cvs.raster(tile, **self.agg_params)
        return agg

    def color_mapping(self, agg):
        img = tf.shade(agg, **self.cm_params)
        return img


class MatplotLibBase(Base):
    def __init__(self, normalization={}, color_mapping={}):
        super().__init__(normalization=normalization, color_mapping=color_mapping)

        self.method = self.norm_params.get("method", None)

        self.method_kwargs = self.norm_params.get("method_kwargs", {})

        if not self.cm_params.get("cm", False):
            self.cm_params = {"cm": cm.viridis}

        self.cmap = cm.get_cmap(self.cm_params["cm"])
        self.cmap_kwargs = self.cm_params.get("cm_kwargs", {})

    def normalization(self, tile):
        if self.method:
            for k, f in self.method_kwargs.items():
                if callable(f):
                    self.method_kwargs[k] = partial(f, tile)()
            norm = self.method(**self.method_kwargs)
            return norm(tile)
        else:
            return tile

    def color_mapping(self, tile):
        arr = self.cmap(tile, **self.cmap_kwargs)
        return Image.fromarray(uint8(arr * 255), "RGBA")
