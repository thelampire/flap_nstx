from .get_data_gpi import get_data_gpi, add_coordinate_gpi
from .get_data_thomson import get_data_thomson, add_coordinate_thomson
#No get_data_chers, avoiding the use of flap

from .get_data import register

from . import gpi
from . import analysis
from . import chers
from . import thomson
from . import publications
from . import tools

import sys

# Add the workspace root to Python path for MDSplus
#workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#if workspace_root not in sys.path:
#    sys.path.insert(0, workspace_root)