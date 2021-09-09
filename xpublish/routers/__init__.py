from .base import BaseFactory
from .common import common_router, dataset_collection_router
from .zarr import ZarrFactory
try:
    from .xyz import XYZFactory
except ImportError:
    pass