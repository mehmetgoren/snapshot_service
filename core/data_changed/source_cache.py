from abc import ABC, abstractmethod
from redis.client import Redis

from common.data.source_model import SourceModel
from common.data.source_repository import SourceRepository
from common.utilities import logger


class BaseCache(ABC):
    @abstractmethod
    def get(self, source_id: str):
        raise NotImplementedError('BaseCache.get')

    @abstractmethod
    def refresh(self, source_id: str):
        raise NotImplementedError('BaseCache.refresh')

    @abstractmethod
    def remove(self, source_id: str):
        raise NotImplementedError('BaseCache.remove')


class SourceCache(BaseCache):
    dic = {}

    def __init__(self, connection: Redis):
        self.source_repository = SourceRepository(connection)

    @staticmethod
    def set_dict(dic: dict):
        SourceCache.dic = dic

    def get(self, source_id: str) -> SourceModel | None:
        if source_id not in SourceCache.dic:
            source_model = self.source_repository.get(source_id)
            if source_model is None:
                logger.warning(f'source was not found for Object Detection Model, Detection will not work for {source_id}')
                return None
            SourceCache.dic[source_id] = source_model
        return SourceCache.dic[source_id]

    def refresh(self, source_id: str) -> SourceModel | None:
        if source_id in SourceCache.dic:
            SourceCache.dic.pop(source_id)
        return self.get(source_id)

    def remove(self, source_id: str):
        SourceCache.dic[source_id] = None
