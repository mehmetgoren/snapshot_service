import datetime
from typing import List
from datetime import timedelta
from shapely.geometry import Polygon

from core.data_changed.od.od_model import OdModel
from core.filters.detections import DetectionBox


class Od:
    def __init__(self):
        self.id: str = ''
        self.brand: str = ''
        self.name: str = ''
        self.address: str = ''
        self.created_at: str = ''
        self.selected_list_length: int = 0
        self.selected_list: dict = {}
        self.zones_list: List[Polygon] = []
        self.masks_list: List[Polygon] = []
        self.start_time: timedelta = timedelta()
        self.end_time: timedelta = timedelta()
        self.time_in_enabled: bool = False
        self.separator = 'º'
        self.array_separator = '+'

    @staticmethod
    def __create_area(do: DetectionBox):
        x1, y1, x2, y2 = do.x1, do.y1, do.x2, do.y2
        return Polygon([(x1, y1), (x1, y2), (x2, y1), (x2, y2)])

    @staticmethod
    def __create_polygon(value: str, separator: str):
        if len(value) == 0:
            return Polygon([])
        arr = value.split(separator)  # arr.length is always an even number
        length = int(len(arr) / 2)
        index = 0
        points = []
        for j in range(length):
            x = float(arr[index])
            index += 1
            y = float(arr[index])
            index += 1
            points.append((x, y))
        return Polygon(points)

    def __create_polygon_list(self, line: str) -> List[Polygon]:
        ret: List[Polygon] = []
        if len(line) > 0:
            ret = []
            splits = line.split(self.array_separator)
            for split in splits:
                zone_list = self.__create_polygon(split, self.separator)
                ret.append(zone_list)
        return ret

    def is_in_zones(self, do: DetectionBox) -> bool:
        if len(self.zones_list) == 0:
            return True
        area = self.__create_area(do)
        for zone_list in self.zones_list:
            if zone_list.length == 0:
                continue
            if zone_list.intersects(area):
                return True
        return False

    def is_in_masks(self, do: DetectionBox) -> bool:
        if len(self.masks_list) == 0:
            return False
        area = self.__create_area(do)
        for mask_list in self.masks_list:
            if mask_list.length == 0:
                continue
            if mask_list.intersects(area):
                return True
        return False

    def is_selected(self, cls_idx: int) -> bool:
        return cls_idx in self.selected_list

    def check_threshold(self, cls_idx: int, threshold: float) -> bool:
        return threshold >= self.selected_list[cls_idx]

    def is_in_time(self) -> bool:
        if not self.time_in_enabled:
            return True
        now = datetime.datetime.now()
        now_time = timedelta(hours=now.hour, minutes=now.minute)
        return self.start_time <= now_time <= self.end_time

    @staticmethod
    def __int_try_parse(value) -> (bool, int):
        try:
            return True, int(value)
        except ValueError:
            return False, value

    @staticmethod
    def __get_time(time_text: str) -> (bool, timedelta):
        if len(time_text) == 0:
            return False, timedelta()
        splits = time_text.split(':')
        if len(splits) != 2:
            return False, timedelta()

        ok, hour = Od.__int_try_parse(splits[0])
        if not ok:
            return False, timedelta()
        ok, minute = Od.__int_try_parse(splits[1])
        if not ok:
            return False, timedelta()
        return True, timedelta(hours=hour, minutes=minute)

    def map_from(self, od_model: OdModel):
        self.id = od_model.id
        self.brand = od_model.brand
        self.name = od_model.name
        self.address = od_model.address
        self.created_at = od_model.created_at
        indices = [int(item) for item in od_model.selected_list.split(self.separator)] if len(od_model.selected_list) > 0 else []
        self.selected_list_length = len(indices)
        if self.selected_list_length > 0:
            thresholds = od_model.threshold_list.split(self.separator)  # thresholds.length must be equal to indices' length
            index = 0
            for cls_idx in indices:
                self.selected_list[cls_idx] = float(thresholds[index])
                index += 1

        self.zones_list = self.__create_polygon_list(od_model.zones_list)
        self.masks_list = self.__create_polygon_list(od_model.masks_list)
        self.time_in_enabled, self.start_time = self.__get_time(od_model.start_time)
        if self.time_in_enabled:
            self.time_in_enabled, self.end_time = self.__get_time(od_model.end_time)

        return self
