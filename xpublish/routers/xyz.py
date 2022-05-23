from dataclasses import dataclass, field
import xarray as xr
import cachey
from fastapi import Depends, Response, Query, Path
import morecantile
import io

from .factory import XpublishFactory
from xpublish.utils.cache import CostTimer
from xpublish.utils.api import DATASET_ID_ATTR_KEY
from xpublish.dependencies import get_dataset, get_cache
from xpublish.utils.ows import (
    get_bounds,
    get_tiles,
    query_builder,
    FieldValidator,
    validate_crs,
    WEB_CRS,
    DataShader,
    Render,
)


@dataclass
class XYZFactory(XpublishFactory):
    r"""Class factory for the XYZ router

    Parameters
    ----------
    crs_epsg : int
        Set the EPSG code according to the xarray.Dataset's coordinate reference system (CRS).
        The EPSG code is used in the ``morecantile`` package to create the tiles.
        Supported EPSGs are:
            3857: "WebMercatorQuad"
            32631: "UTM31WGS84Quad"
            3978: "CanadianNAD83_LCC"
            5482: "LINZAntarticaMapTilegrid"
            4326: "WorldCRS84Quad"
            5041: "UPSAntarcticWGS84Quad"
            3035: "EuropeanETRS89_LAEAQuad"
            3395: "WorldMercatorWGS84Quad"
            2193: "NZTM2000"
    transformers : list, optional
        List of callback functions to perform transformation on arrays (datasets or tiles).
        The function name must be either `transform0`, `transform1` or `transform2`.
        Th function name correspond to the point in the workflow where the transformation is applied:
            transform0 -> applied to the whole dataset
            transform1 -> applied to each tile before rendering
            transform2 -> applied to each tile before converting to byte image
    render : :class:`Render`
        The class that configures the rendering of the tiles.
        There are two subclasses available, that correspond to the plotting library:
        `DataShader` (default) and `MatplotLib`.
        TODO: configuration

    Returns
    -------

    Raises
    ------

    Examples
    --------
    transformer example

    DataShader render example

    MatplotLib render example
    """

    crs_epsg: int = FieldValidator(default=4326, validators=(validate_crs,))

    transformers: list = field(default_factory=lambda: [])

    trsf_names: list = field(default_factory=lambda: [], init=False)

    render: Render = field(default=None)

    def __post_init__(self):

        super().__post_init__()

        for t in self.transformers:
            self.trsf_names.append(t.__name__)
            setattr(self, t.__name__, t)

        if self.render is None:
            self.render = DataShader(
                aggregation={}, color_mapping={"cmap": ["blue", "red"]}
            )

    def register_routes(self):
        @self.router.get("/tiles/{var}/{z}/{x}/{y}")
        @self.router.get("/tiles/{var}/{z}/{x}/{y}.{format}")
        async def tiles(
            var: str = Path(
                ..., description="Dataset's variable. It defines the map's data layer"
            ),
            z: int = Path(..., description="Tiles' zoom level"),
            x: int = Path(..., description="Tiles' column"),
            y: int = Path(..., description="Tiles' row"),
            format: str = Query("PNG", description="Image format. Default to PNG"),
            time: str = Query(
                None,
                description="Filter by time in time-varying datasets. String time format should match dataset's time format",
            ),
            xlab: str = Query("x", description="Dataset x coordinate label"),
            ylab: str = Query("y", description="Dataset y coordinate label"),
            cache: cachey.Cache = Depends(get_cache),
            dataset: xr.Dataset = Depends(get_dataset),
        ):

            TMS = morecantile.tms.get(WEB_CRS[self.crs_epsg])

            xleft, xright, ybottom, ytop = get_bounds(TMS, z, x, y)

            query = query_builder(time, xleft, xright, ybottom, ytop, xlab, ylab)

            cache_key = (
                dataset.attrs.get(DATASET_ID_ATTR_KEY, "")
                + "/"
                + f"/tiles/{var}/{z}/{x}/{y}.{format}?{time}"
            )
            response = cache.get(cache_key)

            if response is None:
                with CostTimer() as ct:

                    # transformer 0: over the whole dataset
                    dataset = self("transform0", dataset)

                    tile = get_tiles(var, dataset, query)

                    # transformer 1: over each individual tile before rendering
                    tile = self("transform1", tile)

                    tile = self.render.interpolation(tile)
                    tile = self.render.aggregation(tile)
                    tile = self.render.normalization(tile)
                    img = self.render.color_mapping(tile)

                    # transformer 2: over each individual tile before saving to image
                    img = self("transform2", img)

                    if self.render.__class__.__name__ == "DataShader":
                        img_io = img.to_bytesio(format)
                        byte_image = img_io.read()
                    else:
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        byte_image = buffer.getvalue()

                    response = Response(
                        content=byte_image, media_type=f"image/{format}"
                    )

                cache.put(cache_key, response, ct.time, len(byte_image))

            return response

    def __call__(self, name, array, *args, **kwargs):
        if name in self.trsf_names:
            f = getattr(self, name)
            arr = f(array, *args, **kwargs)
            return arr
        return array
