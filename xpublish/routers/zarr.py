import json
import logging
from dataclasses import dataclass

import cachey
import xarray as xr
from fastapi import Depends, HTTPException
from starlette.responses import Response
from zarr.storage import array_meta_key, attrs_key, group_meta_key

from ..dependencies import get_cache, get_dataset, get_zmetadata, get_zvariables
from ..utils.api import DATASET_ID_ATTR_KEY
from ..utils.cache import CostTimer
from ..utils.zarr import encode_chunk, get_data_chunk, jsonify_zmetadata, zarr_metadata_key
from .factory import XpublishFactory

logger = logging.getLogger('zarr_api')


@dataclass
class ZarrFactory(XpublishFactory):
    """Provides access to data and metadata through as Zarr compatible API."""

    def register_routes(self):
        @self.router.get(f'/{zarr_metadata_key}')
        def get_zarr_metadata(
            dataset=Depends(self.dataset_dependency),
            cache=Depends(self.cache_dependency),
        ):
            zvariables = get_zvariables(dataset, cache)
            zmetadata = get_zmetadata(dataset, cache, zvariables)

            zjson = jsonify_zmetadata(dataset, zmetadata)

            return Response(json.dumps(zjson).encode('ascii'), media_type='application/json')

        @self.router.get(f'/{group_meta_key}')
        def get_zarr_group(
            dataset=Depends(self.dataset_dependency),
            cache=Depends(self.cache_dependency),
        ):
            zvariables = get_zvariables(dataset, cache)
            zmetadata = get_zmetadata(dataset, cache, zvariables)

            return zmetadata['metadata'][group_meta_key]

        @self.router.get(f'/{attrs_key}')
        def get_zarr_attrs(
            dataset=Depends(self.dataset_dependency),
            cache=Depends(self.cache_dependency),
        ):
            zvariables = get_zvariables(dataset, cache)
            zmetadata = get_zmetadata(dataset, cache, zvariables)

            return zmetadata['metadata'][attrs_key]

        @self.router.get('/{var}/{chunk}')
        def get_variable_chunk(
            var: str,
            chunk: str,
            dataset: xr.Dataset = Depends(get_dataset),
            cache: cachey.Cache = Depends(get_cache),
        ):
            """Get a zarr array chunk.

            This will return cached responses when available.

            """
            zvariables = get_zvariables(dataset, cache)
            zmetadata = get_zmetadata(dataset, cache, zvariables)

            # First check that this request wasn't for variable metadata
            if array_meta_key in chunk:
                return zmetadata['metadata'][f'{var}/{array_meta_key}']
            elif attrs_key in chunk:
                return zmetadata['metadata'][f'{var}/{attrs_key}']
            elif group_meta_key in chunk:
                raise HTTPException(status_code=404, detail='No subgroups')
            else:
                logger.debug('var is %s', var)
                logger.debug('chunk is %s', chunk)

                cache_key = dataset.attrs.get(DATASET_ID_ATTR_KEY, '') + '/' + f'{var}/{chunk}'
                response = cache.get(cache_key)

                if response is None:
                    with CostTimer() as ct:
                        arr_meta = zmetadata['metadata'][f'{var}/{array_meta_key}']
                        da = zvariables[var].data

                        data_chunk = get_data_chunk(da, chunk, out_shape=arr_meta['chunks'])

                        echunk = encode_chunk(
                            data_chunk.tobytes(),
                            filters=arr_meta['filters'],
                            compressor=arr_meta['compressor'],
                        )

                        response = Response(echunk, media_type='application/octet-stream')

                    cache.put(cache_key, response, ct.time, len(echunk))

                return response
