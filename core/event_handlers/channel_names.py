from enum import Enum


class EventChannels(str, Enum):
    read_service = 'read_service'
    data_changed = 'data_changed'
    snapshot_in = 'snapshot_in'
    snapshot_out = 'snapshot_out'
    od_service = 'od_service'
