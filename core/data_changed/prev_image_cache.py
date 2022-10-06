from typing import Any


class PrevImageCache:
    def __init__(self, dic: dict):
        self.dic = dic

    def get(self, source_id: str) -> Any | None:
        if not self.has(source_id):
            return None
        return self.dic[source_id]

    def has(self, source_id: str) -> bool:
        return source_id in self.dic

    def set(self, source_id: str, prev_image: any):
        self.dic[source_id] = prev_image
