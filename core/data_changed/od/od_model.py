from common.data.source_model import SourceModel


class OdModel:
    def __init__(self):
        self.id: str = ''
        self.brand: str = ''
        self.name: str = ''
        self.address: str = ''
        self.created_at: str = ''
        self.threshold_list: str = '0.8'
        self.selected_list: str = '0'
        self.zones_list: str = ''
        self.masks_list: str = ''
        self.start_time: str = ''
        self.end_time: str = ''

    def map_from(self, source: SourceModel):
        self.id = source.id
        self.brand = source.brand
        self.name = source.name
        self.address = source.address
        return self
