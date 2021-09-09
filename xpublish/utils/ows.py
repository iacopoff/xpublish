from datashader import transfer_functions as tf
import datashader as ds
import xarray as xr
from fastapi import HTTPException
import morecantile


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
        raise DataValidationError(f"User input {crs_epsg} not supported")


class LayerOptionsMixin:
    def set_options(self, crs_epsg: int = 3857, color_mapping: dict = {}) -> None:

        self.datashader_settings = color_mapping.get("datashader_settings")
        self.matplotlib_settings = color_mapping.get("matplotlib_settings")

        if crs_epsg not in WEB_CRS.keys():
            raise DataValidationError(f"User input {crs_epsg} not supported")

        self.TMS = morecantile.tms.get(WEB_CRS[crs_epsg])


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

    if query.get("time"):
        tile = dataset[var].sel(query)  # noqa
    else:
        tile = dataset[var].sel(query)  # noqa

    if 0 in tile.sizes.values():
        raise HTTPException(status_code=406, detail=f"Map outside dataset domain")

    return tile


def get_image_datashader(tile, datashader_settings, format):

    raster_param = datashader_settings.get("raster", {})
    shade_param = datashader_settings.get("shade", {"cmap": ["blue", "red"]})

    cvs = ds.Canvas(plot_width=256, plot_height=256)

    agg = cvs.raster(tile, **raster_param)

    img = tf.shade(agg, **shade_param)

    img_io = img.to_bytesio(format)

    return img_io.read()


def get_legend():
    pass


def validate_dataset(dataset):
    dims = dataset.dims
    if "x" not in dims or "y" not in dims:
        raise DataValidationError(
            f" Expected spatial dimension names 'x' and 'y', found: {dims}"
        )
    if "time" not in dims and len(dims) >= 3:
        raise DataValidationError(
            f" Expected time dimension name 'time', found: {dims}"
        )
    if len(dims) > 4:
        raise DataValidationError(f" Not implemented for dimensions > 4")
