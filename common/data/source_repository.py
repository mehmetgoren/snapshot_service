from redis.client import Redis

from common.data.base_repository import BaseRepository
from common.data.source_model import SourceModel


class SourceRepository(BaseRepository):
    def __init__(self, connection: Redis):
        super().__init__(connection, 'sources:')

    def _get_key(self, identifier: str) -> str:
        return f'{self.namespace}{identifier}'

    def get(self, identifier: str) -> SourceModel:
        key = self._get_key(identifier)
        dic = self.connection.hgetall(key)
        if not dic:
            return None
        return self.from_redis(SourceModel(), dic)
