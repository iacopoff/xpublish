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
    DataShaderBase,
)


@dataclass
class XYZFactory(XpublishFactory):

    crs_epsg: int = FieldValidator(default=4326, validators=(validate_crs,))

    transformers: list = field(default_factory=lambda: [])

    trsf_names: list = field(default_factory=lambda: [], init=False)

    renderer: DataShaderBase = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        for t in self.transformers:
            self.trsf_names.append(t.__name__)
            setattr(self, t.__name__, t)

        if self.renderer is None:
            self.renderer = DataShaderBase(
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

            # color mapping settings
            # datashader_settings = self.color_mapping.get("datashader_settings")

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
                    if self("transform0", dataset):
                        return

                    tile = get_tiles(var, dataset, query)

                    # transformer 1: over each individual tile
                    if self("transform1", tile):
                        return

                    tile = self.renderer.interpolation(tile)
                    tile = self.renderer.aggregation(tile)
                    tile = self.renderer.normalization(tile)
                    img = self.renderer.color_mapping(tile)

                    if self.renderer.__class__.__name__ == "DataShader":
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
            if f(array, *args, **kwargs):
                return True
        return False
