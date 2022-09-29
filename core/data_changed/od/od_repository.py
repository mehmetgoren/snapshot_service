from redis.client import Redis

from common.data.base_repository import BaseRepository
from common.utilities import datetime_now
from core.data_changed.od.od_model import OdModel


class OdRepository(BaseRepository):
    def __init__(self, connection: Redis):
        super().__init__(connection, 'ods:')

    def _get_key(self, key: str):
        return f'{self.namespace}{key}'

    def add(self, model: OdModel) -> int:
        key = self._get_key(model.id)
        model.created_at = datetime_now()
        dic = self.to_redis(model)
        return self.connection.hset(key, mapping=dic)

    def get(self, identifier: str) -> OdModel:
        key = self._get_key(identifier)
        dic = self.connection.hgetall(key)
        if not dic:
            return None
        model: OdModel = self.from_redis(OdModel(), dic)
        return model
