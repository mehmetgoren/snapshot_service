from redis.client import Redis

from common.utilities import logger
from core.data_changed.od.od import Od
from core.data_changed.od.od_model import OdModel
from core.data_changed.od.od_repository import OdRepository
from core.data_changed.source.source_cache import SourceCache, BaseCache


class OdCache(BaseCache):
    models = {}

    def __init__(self, connection: Redis, source_cache: SourceCache):
        self.od_repository = OdRepository(connection)
        self.source_cache = source_cache

    def get(self, source_id: str) -> Od | None:
        if source_id not in OdCache.models:
            od_model = self.od_repository.get(source_id)
            if od_model is None:
                source_model = self.source_cache.get(source_id)
                if source_model is None:
                    logger.warning(f'source was not found for Object Detection Model, Detection will not work for {source_id}')
                    return None
                od_model = OdModel().map_from(source_model)
                self.od_repository.add(od_model)
            OdCache.models[source_id] = Od().map_from(od_model)
        return OdCache.models[source_id]

    def refresh(self, source_id: str) -> Od | None:
        if source_id in OdCache.models:
            del OdCache.models[source_id]
        return self.get(source_id)

    def remove(self, source_id: str):
        OdCache.models[source_id] = None
