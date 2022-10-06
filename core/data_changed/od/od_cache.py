from redis.client import Redis

from common.utilities import logger
from core.data_changed.od.od import Od
from core.data_changed.od.od_model import OdModel
from core.data_changed.od.od_repository import OdRepository
from core.data_changed.source_cache import BaseCache, SourceCache


class OdCache(BaseCache):
    dic = {}

    def __init__(self, connection: Redis, source_cache: SourceCache):
        self.od_repository = OdRepository(connection)
        self.source_cache = source_cache

    @staticmethod
    def set_dict(dic: dict):
        OdCache.dic = dic

    def get(self, source_id: str) -> Od | None:
        if source_id not in OdCache.dic:
            od_model = self.od_repository.get(source_id)
            if od_model is None:
                source_model = self.source_cache.get(source_id)
                if source_model is None:
                    logger.warning(f'source was not found for Object Detection Model, Detection will not work for {source_id}')
                    return None
                od_model = OdModel().map_from(source_model)
                self.od_repository.add(od_model)
            OdCache.dic[source_id] = Od().map_from(od_model)
        return OdCache.dic[source_id]

    def refresh(self, source_id: str) -> Od | None:
        if source_id in OdCache.dic:
            OdCache.dic.pop(source_id)
        return self.get(source_id)

    def remove(self, source_id: str):
        OdCache.dic[source_id] = None
