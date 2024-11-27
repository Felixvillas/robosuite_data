from robosuite.environments.base import make

# Manipulation environments
from robosuite.environments.manipulation.jimu import Jimu
from robosuite.environments.manipulation.drawer import Drawer
from robosuite.environments.manipulation.mstt import Mstt

from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.nut_assembly import NutAssembly
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.environments.manipulation.door import Door
from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments.manipulation.tool_hang import ToolHang
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
from robosuite.environments.manipulation.two_arm_handover import TwoArmHandover
from robosuite.environments.manipulation.two_arm_transport import TwoArmTransport
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from robosuite.environments.manipulation.jimu import Jimu
from robosuite.environments.manipulation.jimu_lift import JimuLift
from robosuite.environments.manipulation.sunmao_jimu import SunmaoJimu
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.robots import ALL_ROBOTS
from robosuite.models.grippers import ALL_GRIPPERS

__version__ = "1.4.1"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
