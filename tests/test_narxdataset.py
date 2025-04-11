
from torchhydro.datasets.narxdataset import NarxDataset

class TestDatasource:
    def __init__(self, source_cfgs, time_unit="1D"):
        self.ngrid = 2
        self.nt = 366
        self.data_cfgs = source_cfgs

def test_read_xyc_specified_time():
    """

    Returns
    -------

    """
