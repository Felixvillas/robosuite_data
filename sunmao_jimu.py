from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, JimuObject, JimuVisualObject, SBoxVisualObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, UniformFixSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat, quat2axisangle
import pdb
from copy import deepcopy
import math, json, itertools

def print_red(msg):
    print("\033[91m {}\033[00m".format(msg))
    
def print_green(msg):
    print("\033[92m {}\033[00m".format(msg))


class SunmaoJimu(SingleArmEnv):
    """
    This class corresponds to the SunmaoJimu task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer
        
        # >>>>>>>>>>>>>>>>>>>>>> some attribute of sunmao_jimu <<<<<<<<<<<<<<<<<<<<<<
        self.num_correct_jimu_objects = 1
        assert self.num_correct_jimu_objects == 1
        
        self.num_objects_per_slot = 2
        assert self.num_objects_per_slot == 2
        
        self.slot_min, self.slot_max = 1, 6
        
        # >>>>>>>>>>>>>>>>>>>>>> some attribute of sunmao_jimu <<<<<<<<<<<<<<<<<<<<<<

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 2.0 if r_stack > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # return 0, 0, 0
        # reaching is successful when the gripper site is close to the center of the cube
        #TODO: for multiple target cubes
        cubeA_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(self.moved_correct_jimu_objects[-1].root_body)]
        cubeA_rotation = quat2axisangle(
            convert_quat(np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(self.moved_correct_jimu_objects[-1].root_body)]), to="xyzw")
        )
        cubeB_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(self.virtual_jimu_objects[-1].root_body)]
        cubeB_rotation = quat2axisangle(
            convert_quat(np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(self.virtual_jimu_objects[-1].root_body)]), to="xyzw")
        )
        
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # grasping reward
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.moved_correct_jimu_objects[-1])
        if grasping_cubeA:
            r_reach += 0.25

        # lifting is successful when the cube is above the table top by a margin
        cubeA_height = cubeA_pos[2]
        table_height = self.table_offset[2]
        cubeA_lifted = cubeA_height > table_height + 0.04
        r_lift = 1.0 if cubeA_lifted else 0.0

        # Aligning is successful when cubeA is right above cubeB and cubeA is parallel to cubeB
        # raise NotImplementedError
        # rotation_cos = np.dot(cubeA_rotation, cubeB_rotation) / (np.linalg.norm(cubeA_rotation) * np.linalg.norm(cubeB_rotation))
        rotation_cos = math.cos((cubeA_rotation[-1] - cubeB_rotation[-1]) % math.pi)
        rotation_cos_abs = abs(rotation_cos)
        if cubeA_lifted:
            horiz_dist = np.linalg.norm(np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2]))
            r_lift += 0.25 * (1 - np.tanh(horiz_dist))
            r_lift += 0.25 * rotation_cos_abs
            

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_stack = 0
        # try:
            
        distance_A_B = np.linalg.norm(np.array(cubeA_pos) - np.array(cubeB_pos))
        
        cubeA_in_right_pos = (distance_A_B + (1 - rotation_cos_abs)) < 0.02
        
        # pdb.set_trace()
        print(f"distance_A_B: {distance_A_B}, rotation_cos_abs: {rotation_cos_abs}")
        # print(f"cubeA_pos: {cubeA_pos}, cubeB_pos: {cubeB_pos}, visual_cube_index: {self.src_cube_id}, visual_cube_B_pos: {visual_cube_B_pos}, distance: {distance_A_B}")
        if not grasping_cubeA and cubeA_in_right_pos:
            r_stack = 2.0
        # except:
        #     breakpoint()

        # pdb.set_trace()
        return r_reach, r_lift, r_stack
    

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        print_red(f"self.deterministic_reset: {self.deterministic_reset}")
        super()._load_model()

        # pdb.set_trace()
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        # >>>>>>>>>>>>>>>>>>>>>> all summao_jimu <<<<<<<<<<<<<<<<<<<<<<
        slot_block_dict = {
            1: {"name": "s1", "jimu_type": "1_slot_block"},
            2: {"name": "s2", "jimu_type": "2_slot_block"},
            3: {"name": "s3", "jimu_type": "3_slot_block"},
            4: {"name": "s4", "jimu_type": "4_slot_block"},
            5: {"name": "s5", "jimu_type": "5_slot_block"},
            6: {"name": "s6", "jimu_type": "6_slot_block"},
        }
        slot_visual_block_dict = {
            1: {"name": "visual_s1", "jimu_type": "1_slot_block"},
            2: {"name": "visual_s2", "jimu_type": "2_slot_block"},
            3: {"name": "visual_s3", "jimu_type": "3_slot_block"},
            4: {"name": "visual_s4", "jimu_type": "4_slot_block"},
            5: {"name": "visual_s5", "jimu_type": "5_slot_block"},
            6: {"name": "visual_s6", "jimu_type": "6_slot_block"},
        }
        
        physical_jimu_objects = [] # the physical sunmao_jimu in the main body
        virtual_jimu_objects = [] # the virtual sunmao_jimu in the main body
        moved_jimu_objects = [] # the physical sunmao_jimu in the moved_jimu_objects: correct here means the jimu should be moved by gripper
        
        slot_min, slot_max = self.slot_min, self.slot_max
        jimu_objects_ys = [
            [-0.24, -0.20],
            [-0.20, -0.16],
            [-0.16, -0.12],
            [-0.12, -0.08],
            [-0.08, -0.04],
            [-0.04, -0.00],
        ]
        y = [-0.25, 0, 0.25]
        x = np.arange(-0.39, 0.39, 0.03).tolist()
        x_y = list(itertools.product(x, y))
        x_y_idx = 0
        # pdb.set_trace()
        
        for idx, slot in enumerate(reversed(range(slot_min, slot_max + 1))):
            # physical_jimu_objects
            # for i in range(self.num_objects_per_slot):
            physical_jimu_objects.insert(
                0, 
                JimuObject(name="physical_" + slot_block_dict[slot]["name"] + f"_{2}", jimu_type=slot_block_dict[slot]["jimu_type"])   
            )
            self.placement_initializer.append_sampler(UniformFixSampler(
                name=f"physical_jimu_objects_slot_{slot}_{1}",
                mujoco_objects=physical_jimu_objects[0],
                # x_range=[0.0, 0.0], # or -0.04 and +0.04
                # y_range=[jimu_objects_ys[idx][0] - 2, jimu_objects_ys[idx][1] - 2],
                x_range=[x_y[x_y_idx][0], x_y[x_y_idx][0]],
                y_range=[x_y[x_y_idx][1], x_y[x_y_idx][1]],
                rotation=math.pi / 2,
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=-0.2
            ))
            x_y_idx += 1
            physical_jimu_objects.insert(
                0, 
                JimuObject(name="physical_" + slot_block_dict[slot]["name"] + f"_{1}", jimu_type=slot_block_dict[slot]["jimu_type"])   
            )
            self.placement_initializer.append_sampler(UniformFixSampler(
                name=f"physical_jimu_objects_slot_{slot}_{2}",
                mujoco_objects=physical_jimu_objects[0],
                # x_range=[0.0, 0.0], # or -0.04 and +0.04
                # y_range=[jimu_objects_ys[idx][0] + 2, jimu_objects_ys[idx][1] + 2],
                x_range=[x_y[x_y_idx][0], x_y[x_y_idx][0]],
                y_range=[x_y[x_y_idx][1], x_y[x_y_idx][1]],
                rotation=math.pi / 2,
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=-0.2
            ))
            x_y_idx += 1
            # virtual_jimu_objects
            # for i in range(self.num_objects_per_slot):
            virtual_jimu_objects.insert(
                0, 
                JimuVisualObject(name="virtual_" + slot_visual_block_dict[slot]["name"], jimu_type=slot_visual_block_dict[slot]["jimu_type"])   
            )
            self.placement_initializer.append_sampler(UniformFixSampler(
                name=f"virtual_jimu_objects_slot_{slot}_{1}",
                mujoco_objects=virtual_jimu_objects[0],
                # x_range=[0.0, 0.0], # or -0.04 and +0.04
                # y_range=[jimu_objects_ys[idx][0] - 3, jimu_objects_ys[idx][1] - 3],
                x_range=[x_y[x_y_idx][0], x_y[x_y_idx][0]],
                y_range=[x_y[x_y_idx][1], x_y[x_y_idx][1]],
                rotation=math.pi / 2,
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=-0.2
            ))
            x_y_idx += 1
            # virtual_jimu_objects.append(
            #     JimuVisualObject(name="virtual_" + slot_visual_block_dict[slot]["name"] + f"_{2}", jimu_type=slot_visual_block_dict[slot]["jimu_type"])   
            # )
            # self.placement_initializer.append_sampler(UniformFixSampler(
            #     name=f"virtual_jimu_objects_slot_{slot}_{2}",
            #     mujoco_objects=virtual_jimu_objects[-1],
            #     # x_range=[0.0, 0.0], # or -0.04 and +0.04
            #     # y_range=[jimu_objects_ys[idx][0] + 3, jimu_objects_ys[idx][1] + 3],
            #     x_range=[x_y[x_y_idx][0], x_y[x_y_idx][0]],
            #     y_range=[x_y[x_y_idx][1], x_y[x_y_idx][1]],
            #     rotation=0,
            #     rotation_axis='z',
            #     ensure_object_boundary_in_range=False,
            #     ensure_valid_placement=True,
            #     reference_pos=self.table_offset,
            #     z_offset=-0.2
            # ))
            # x_y_idx += 1
            
        print_red(f"x_y_idx: {x_y_idx}")
        for idx, slot in enumerate(reversed(range(slot_min, slot_max + 1))):
            # moved_correct_jimu_objects && moved_incorrect_jimu_objects
            moved_jimu_objects.insert(
                0, 
                JimuObject(name="moved_" + slot_block_dict[slot]["name"], jimu_type=slot_block_dict[slot]["jimu_type"])   
            )
            self.placement_initializer.append_sampler(UniformFixSampler(
                name=f"moved_jimu_objects_slot_{slot}",
                mujoco_objects=moved_jimu_objects[0],
                # x_range=[0.0, 0.0], # or -0.04 and +0.04
                # y_range=[jimu_objects_ys[idx][0] - 4, jimu_objects_ys[idx][1] - 4],
                x_range=[x_y[x_y_idx][0], x_y[x_y_idx][0]],
                y_range=[x_y[x_y_idx][1], x_y[x_y_idx][1]],
                rotation=math.pi / 2,
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=-0.2
            ))
            x_y_idx += 1

        print_red(f"x_y_idx: {x_y_idx}")
        
        # >>>>>>>>>>>>>>>>>>>>>> all summao_jimu <<<<<<<<<<<<<<<<<<<<<<
        self.objs = {
            "physical_jimu_objects": physical_jimu_objects,
            "virtual_jimu_objects": virtual_jimu_objects, 
            "moved_jimu_objects": moved_jimu_objects
        } 
        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=physical_jimu_objects + virtual_jimu_objects + moved_jimu_objects 
        )
    
    def _sunmao_apperance(self):
        # >>>>>>>>>>>>>>>>>>>>>> sample for summao_jimu appearance <<<<<<<<<<<<<<<<<<<<<<
        
        # sample for the number of layer
        layer_min, layer_max = 2, 6
        layer_num = np.random.randint(layer_min, layer_max + 1)
        # layer_num = 2
        # sample for the slot of each layer: The number of slots gradually decreases from bottom to top along the z-axis
        layer_slot_dict = {i: None for i in range(1, layer_num + 1)}
        
        slot_min, slot_max = 1, 6
        for layer in layer_slot_dict.keys():
            if layer == 1:
                slot_true = np.random.randint(layer_num, slot_max + 1)
            else:
                assert layer_slot_dict[layer - 1] is not None
                slot_true = np.random.randint(layer_num - (layer - 1), layer_slot_dict[layer - 1])
            
            # if layer == layer_num:
            #     slot_true = np.random.randint(slot_min, slot_max)
            # else:
            #     # mid and down layer cannot be 1 slot
            #     slot_true = np.random.randint(slot_min + 1, slot_max)
            layer_slot_dict[layer] = slot_true
            
        # layer_slot_dict = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
        # layer_slot_dict = {1: 6, 2: 6}
        
        assert len(layer_slot_dict.keys()) == layer_num
        
        layer_rotation_dict = {i : None for i in layer_slot_dict.keys()}
        for layer in layer_rotation_dict.keys():
            if layer % 2 != 0:
                layer_rotation_dict[layer] = 0
            else:
                layer_rotation_dict[layer] = math.pi / 2 # or pi / 2?
                
        # just care about the length size, do not care about the width size and height size
        length_left, length_mid, length_right = 0.00425 * 2, 0.0115 * 2, 0.0085 * 2
        layer_long_dict = {
            layer: length_left * 2 + length_mid * slot + length_right * (slot - 1) - length_left * 2 - length_mid / 2 * 2 
            # layer: length_left * 2 + length_mid * slot + length_right * (slot - 1)
            for layer, slot in layer_slot_dict.items()
        }
        
        layer_distance_dict = {layer: None for layer in layer_long_dict.keys()}
        for layer in reversed(layer_long_dict.keys()):
            # if layer == len(layer_distance_dict.keys()):
            if layer == layer_num:
                # do not consider the distance of the two summao_jimu at the top layer
                # Because they don't need to carry the upper level sunmao_jimu,
                # we default to placing them in the two farthest slots on the next level
                continue
            # long_of_the_upper_layer = layer_long_dict[layer + 1]
            # if long_of_the_upper_layer == 0:
            #     # if the upper layer is the 1-slot summao_jimu, 
            #     continue
            
            # layer_distance_dict[layer] = long_of_the_upper_layer
            
            long_of_the_upper_layer = layer_long_dict[layer + 1]
            if layer == 1:
                long_of_the_down_layer = None
                if long_of_the_upper_layer == 0:
                    continue
                else:
                    layer_distance_dict[layer] = long_of_the_upper_layer
            else:
                long_of_the_down_layer = layer_long_dict[layer - 1]
                if long_of_the_upper_layer == 0:
                    # as we select the min between long_of_the_upper_layer and long_of_the_down_layer
                    # so if any of them is 0, we will set the layer_distance_dict as None
                    # None will be process specially
                    # continue
                    layer_distance_dict[layer] = long_of_the_down_layer
                else:
                    layer_distance_dict[layer] = min(long_of_the_upper_layer, long_of_the_down_layer)
            
        layer_xyz_dict = {layer: {} for layer in layer_slot_dict.keys()}
        height = 0.02 * 2
        delta_unit = length_mid + length_right
        for layer in layer_xyz_dict.keys():
            if layer == 1:
                next_layer_center = {"x": 0, "y": 0}
            else:
                next_layer_center = {
                    "x": np.mean(layer_xyz_dict[layer - 1]["x"]), 
                    "y": np.mean(layer_xyz_dict[layer - 1]["y"])
                }
            
            if layer_distance_dict[layer] is None:
                if layer == 1:
                    # layer == 1 and layer_distance_dict[1] == None means the jimu is 2 layers and the top layer is 1 slot
                    # so there is no require for the distance of the 2 jimu in the 1st layer
                    # and defaultly, we set the distance of 2 jimu in the 1st layer as 0.04 
                    if layer_rotation_dict[layer] == 0:
                        layer_xyz_dict[layer]["x"] = [0 + next_layer_center["x"], 0 + next_layer_center["x"]]
                        layer_xyz_dict[layer]["y"] = [-0.02 + next_layer_center["y"], 0.02 + next_layer_center["y"]]
                    else:
                        layer_xyz_dict[layer]["y"] = [0 + next_layer_center["y"], 0 + next_layer_center["y"]]
                        layer_xyz_dict[layer]["x"] = [-0.02 + next_layer_center["x"], 0.02 + next_layer_center["x"]]
                else:
                    # layer != 1 and layer_distance_dict[1] == None <==> (layer == top_layer)
                    # so we need to refer to the down layer to set the (x, y) of this layer
                    if layer_distance_dict[layer - 1] is None:
                        # summao_jimu is 2 layers and the top_layer is 1 slot
                        delta = 0
                    else:
                        num_delta_unit = (layer_long_dict[layer] - layer_distance_dict[layer - 1]) / delta_unit
                        assert abs(num_delta_unit - round(num_delta_unit)) < 1e-8
                        num_delta_unit = round(num_delta_unit)
                        if num_delta_unit % 2:
                            delta = -delta_unit / 2
                        else:
                            delta = 0
                            
                        
                    if layer_rotation_dict[layer] == 0:
                        layer_xyz_dict[layer]["x"] = [0 + next_layer_center["x"] + delta, 0 + next_layer_center["x"] + delta]
                        layer_xyz_dict[layer]["y"] = [-layer_long_dict[layer - 1] / 2 + next_layer_center["y"], layer_long_dict[layer - 1] / 2 + next_layer_center["y"]]
                    else:
                        layer_xyz_dict[layer]["y"] = [0 + next_layer_center["y"] + delta, 0 + next_layer_center["y"] + delta]
                        layer_xyz_dict[layer]["x"] = [-layer_long_dict[layer - 1] / 2 + next_layer_center["x"], layer_long_dict[layer - 1] / 2 + next_layer_center["x"]]
            else:
                # delta by the down layer
                if layer == 1:
                    #### X, Y here is not corresponds to X and Y axis, they are just mean 1st or 2nd
                    delta_X = 0 # do not delta for the lowest layer
                    delta_Y = 0 # do not delta for the lowest layer
                else:
                    num_delta_unit = (layer_long_dict[layer] - layer_distance_dict[layer - 1]) / delta_unit
                    assert abs(num_delta_unit - round(num_delta_unit)) < 1e-8
                    num_delta_unit = round(num_delta_unit)
                    if num_delta_unit % 2:
                        delta_X = -delta_unit / 2
                    else:
                        delta_X = 0
                    
                    num_delta_unit = layer_distance_dict[layer] / delta_unit
                    assert abs(num_delta_unit - round(num_delta_unit)) < 1e-8
                    num_delta_unit = round(num_delta_unit)
                    
                    if num_delta_unit % 2 != layer_slot_dict[layer - 1] % 2:
                        delta_Y = 0
                    else:
                        delta_Y = -delta_unit / 2
                        
                if layer_rotation_dict[layer] == 0:
                    layer_xyz_dict[layer]["x"] = [0 + next_layer_center["x"] + delta_X, 0 + next_layer_center["x"] + delta_X]
                    layer_xyz_dict[layer]["y"] = [-layer_distance_dict[layer] / 2 + delta_Y + next_layer_center["y"], layer_distance_dict[layer] / 2 + delta_Y + next_layer_center["y"]]
                else:
                    layer_xyz_dict[layer]["y"] = [0 + next_layer_center["y"] + delta_X, 0 + next_layer_center["y"] + delta_X]
                    layer_xyz_dict[layer]["x"] = [-layer_distance_dict[layer] / 2 + delta_Y + next_layer_center["x"], layer_distance_dict[layer] / 2 + delta_Y + next_layer_center["x"]]
            
            layer_xyz_dict[layer]["z"] = height / 2 * layer - height / 2 
            
            # Special processing when slot is set to 1
            if layer_slot_dict[layer] == 1:
                if layer_rotation_dict[layer] == 0:
                    layer_xyz_dict[layer]["x"] = layer_xyz_dict[layer - 1]["x"]
                else:
                    layer_xyz_dict[layer]["y"] = layer_xyz_dict[layer - 1]["y"]
            

        # print("layer_slot_dict", layer_slot_dict)
        # print("layer_long_dict", layer_long_dict)
        # print("layer_distance_dict", layer_distance_dict)
        # print("layer_xyz_dict", layer_xyz_dict)
        init_x_range = [0.1, 0.2]
        init_y_range = [0.1, 0.2]
        init_x_delta = np.random.uniform(low=init_x_range[0], high=init_x_range[1])
        init_y_delta = np.random.uniform(low=init_y_range[0], high=init_y_range[1])
        # init_x_delta = 0.
        # init_y_delta = 0.
        
        for layer in layer_xyz_dict.keys():
            xs, ys = layer_xyz_dict[layer]["x"], layer_xyz_dict[layer]["y"]
            layer_xyz_dict[layer]["x"] = [xs[0] + init_x_delta, xs[1] + init_x_delta]
            layer_xyz_dict[layer]["y"] = [ys[0] + init_y_delta, ys[1] + init_y_delta]
        
        self.physical_jimu_objects = [] # the physical sunmao_jimu in the main body
        self.virtual_jimu_objects = [] # the virtual sunmao_jimu in the main body
        self.moved_correct_jimu_objects = [] # the correct physical sunmao_jimu in the moved_jimu_objects: correct here means the jimu should be moved by gripper
        self.moved_incorrect_jimu_objects = [] # the incorrect physical sunmao_jimu that can be moved by gripper
        
        physical_jimu_names = []
        virtual_jimu_names = []
        moved_jimu_names = []
        # print_green(layer_slot_dict)
        # print_green(f"{json.dumps(layer_xyz_dict)}")
        for layer, slot in reversed(layer_slot_dict.items()):
            self.physical_jimu_objects.append(
                self.objs["physical_jimu_objects"][self.num_objects_per_slot * (slot - 1)]
            )
            physical_jimu_names.append(f"physical_jimu_objects_slot_{slot}_1")
            if layer == layer_num:
                self.virtual_jimu_objects.append(
                    # self.objs["virtual_jimu_objects"][self.num_objects_per_slot * (slot - 1)]
                    self.objs["virtual_jimu_objects"][slot - 1]
                )
                virtual_jimu_names.append(f"virtual_jimu_objects_slot_{slot}_1")
            else:
                self.physical_jimu_objects.append(
                    self.objs["physical_jimu_objects"][self.num_objects_per_slot * (slot - 1) + 1]
                )
                physical_jimu_names.append(f"physical_jimu_objects_slot_{slot}_2")
            
            if layer != layer_num:
                # for i in range(self.num_objects_per_slot):
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{1}"].x_range = [layer_xyz_dict[layer]["x"][0], layer_xyz_dict[layer]["x"][0]]
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{1}"].y_range = [layer_xyz_dict[layer]["y"][0], layer_xyz_dict[layer]["y"][0]]
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{1}"].z_offset = layer_xyz_dict[layer]["z"]
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{1}"].rotation = layer_rotation_dict[layer]
                
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{2}"].x_range = [layer_xyz_dict[layer]["x"][1], layer_xyz_dict[layer]["x"][1]]
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{2}"].y_range = [layer_xyz_dict[layer]["y"][1], layer_xyz_dict[layer]["y"][1]]
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{2}"].z_offset = layer_xyz_dict[layer]["z"]
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{2}"].rotation = layer_rotation_dict[layer]
            else:
                # for i in range(self.num_objects_per_slot - 1):
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{1}"].x_range = [layer_xyz_dict[layer]["x"][0], layer_xyz_dict[layer]["x"][0]]
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{1}"].y_range = [layer_xyz_dict[layer]["y"][0], layer_xyz_dict[layer]["y"][0]]
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{1}"].z_offset = layer_xyz_dict[layer]["z"]
                self.placement_initializer.samplers[f"physical_jimu_objects_slot_{slot}_{1}"].rotation = layer_rotation_dict[layer]
                
                # i = self.num_objects_per_slot - 1
                self.placement_initializer.samplers[f"virtual_jimu_objects_slot_{slot}_{1}"].x_range = [layer_xyz_dict[layer]["x"][1], layer_xyz_dict[layer]["x"][1]]
                self.placement_initializer.samplers[f"virtual_jimu_objects_slot_{slot}_{1}"].y_range = [layer_xyz_dict[layer]["y"][1], layer_xyz_dict[layer]["y"][1]]
                self.placement_initializer.samplers[f"virtual_jimu_objects_slot_{slot}_{1}"].z_offset = layer_xyz_dict[layer]["z"]
                self.placement_initializer.samplers[f"virtual_jimu_objects_slot_{slot}_{1}"].rotation = layer_rotation_dict[layer]
            
        #### object to be moved
        # num_of_moved_object = np.random.randint(slot_min, slot_max + 1)
        num_of_moved_object = 3 # do not random for the constant length of symbolic obs, but, ugly. May be improved in the future
        assert num_of_moved_object <= slot_max
        correct_slot = layer_slot_dict[layer_num]
        slots = [1, 2, 3, 4, 5, 6]
        slots.remove(correct_slot) # remove the correct_slot
        moved_objects = np.random.choice(np.array(slots), size=num_of_moved_object - 1, replace=False).tolist()
        # moved_objects = [3, 5]
        moved_objects.append(correct_slot)
        np.random.shuffle(moved_objects)
        
        y_ranges=[
            [-0.36, -0.24],
            # [-0.28, -0.24],
            [-0.24, -0.12],
            # [-0.20, -0.16],
            [-0.12, 0.00],
            # [-0.12, -0.08],
        ]
        
        # print(moved_objects)
        self.moved_correct_jimu_types = [
            np.eye(slot_max)[correct_slot - 1].tolist()
        ]
        self.moved_incorrect_jimu_types = []
        for idx, slot in enumerate(moved_objects):
            moved_jimu_names.append(f"moved_jimu_objects_slot_{slot}")
            if slot == correct_slot:
                self.moved_correct_jimu_objects.append(
                    self.objs["moved_jimu_objects"][slot - 1]
                )
            else:
                self.moved_incorrect_jimu_objects.append(
                    self.objs["moved_jimu_objects"][slot - 1]
                )
                self.moved_incorrect_jimu_types.append(
                    np.eye(slot_max)[slot - 1].tolist()
                )
            self.placement_initializer.samplers[f"moved_jimu_objects_slot_{slot}"].x_range = [-0.1, -0.1]
            self.placement_initializer.samplers[f"moved_jimu_objects_slot_{slot}"].y_range = y_ranges[idx]
            self.placement_initializer.samplers[f"moved_jimu_objects_slot_{slot}"].z_offset = 0.01
            self.placement_initializer.samplers[f"moved_jimu_objects_slot_{slot}"].rotation = 0
        
        # >>>>>>>>>>>>>>>>>>>>>> sample for summao_jimu appearance <<<<<<<<<<<<<<<<<<<<<<
        
        # now only compatiable with moved 1 object
        assert len(self.virtual_jimu_objects) == self.num_correct_jimu_objects\
            and len(self.moved_correct_jimu_objects) == self.num_correct_jimu_objects\
                and len(self.moved_correct_jimu_types) == self.num_correct_jimu_objects
        
        # self.instance_objs_name = [item.name for item in self.physical_jimu_objects + self.virtual_jimu_objects + self.moved_correct_jimu_objects + self.moved_incorrect_jimu_objects]
        self.instance_objs_name = physical_jimu_names + virtual_jimu_names + moved_jimu_names
        print_green(f"slots: {layer_slot_dict}, instance_objs_name: {self.instance_objs_name}")


    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        #self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        #self.cubeB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)
        self.obj_body_id = {}
        self.obj_geom_id = {}
        for obj in self.objs["physical_jimu_objects"] + self.objs["virtual_jimu_objects"] + self.objs["moved_jimu_objects"]:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        
        if self.deterministic_reset:
            # raise NotImplementedError
            print_red(f"deterministic_reset == True")
            self._reset_internal_ckpt()
            self._load_ckpt()
            return

        
        self._sunmao_apperance()
        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        # pdb.set_trace()
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()
            # pdb.set_trace()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # if obj.name not in self.instance_objs_name:
                #     continue
                # pdb.set_trace()
                #self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                if "visual" in obj.name.lower():
                    """
                    ATTENTION: JimuVisualObject can use this branch; JimuObject cannot use this branch, 
                    will make self.sim.data.body_xpos and self.sim.data.body_xquat
                    """
                    self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                    self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                else:
                    """
                    ATTENTION: JimuObject can use this branch; JimuVisualObject cannot use this branch, 
                    will make IndexError: list index out of range
                    """
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        # pdb.set_trace()
    
    def _load_ckpt(self):
        
        ############## variable should be reset ##############
        self.physical_jimu_objects = []
        self.virtual_jimu_objects = []
        self.moved_correct_jimu_objects, self.moved_correct_jimu_types = [], []
        self.moved_incorrect_jimu_objects, self.moved_incorrect_jimu_types = [], []
        ############## variable should be reset ##############
        
        
        ############## ugly, maybe should be improved ##############
        start_idx = 23
        physical_jimu_idx = 23 + 12
        virtual_jimu_idx = 23 + 12 + 6
        moved_jimu_idx = 23 + 12 + 6 + 6
        assert moved_jimu_idx == self.sim.data.body_xpos.shape[0]
        table_z = 0.8
        delta_z_error = 0.01
        table_z -= delta_z_error
        
        # physical_jimu
        physical_jimu_xpos = self.sim.data.body_xpos[start_idx: physical_jimu_idx]
        physical_jimu_idxs = np.where(physical_jimu_xpos[:, -1] > table_z)[0]
        self.physical_jimu_objects = [self.objs["physical_jimu_objects"][idx] for idx in physical_jimu_idxs]
        
        # virtual_jimu
        virtual_jimu_xpos = self.sim.data.body_xpos[physical_jimu_idx: virtual_jimu_idx]
        virtual_jimu_idxs = np.where(virtual_jimu_xpos[:, -1] > table_z)[0]
        self.virtual_jimu_objects = [self.objs["virtual_jimu_objects"][idx] for idx in virtual_jimu_idxs]
        
        # moved_jimu
        moved_jimu_xpos = self.sim.data.body_xpos[virtual_jimu_idx: moved_jimu_idx]
        moved_jimu_idxs = np.where(moved_jimu_xpos[:, -1] > table_z)[0]
        moved_correct_jimu_idxs = [idx for idx in moved_jimu_idxs if idx in virtual_jimu_idxs]
        moved_incorrect_jimu_idxs = [idx for idx in moved_jimu_idxs if idx not in moved_correct_jimu_idxs]
        self.moved_correct_jimu_objects = [self.objs["moved_jimu_objects"][idx] for idx in moved_correct_jimu_idxs]
        self.moved_correct_jimu_types = [np.eye(self.slot_max)[idx] for idx in moved_correct_jimu_idxs]
        self.moved_incorrect_jimu_objects = [self.objs["moved_jimu_objects"][idx] for idx in moved_incorrect_jimu_idxs]
        self.moved_incorrect_jimu_types = [np.eye(self.slot_max)[idx] for idx in moved_incorrect_jimu_idxs]
        ############## ugly, maybe should be improved ##############
        return

        
    
    def _reset_internal_ckpt(self):
                
        self.sim.forward()
        # self._record_target_infos()
        # pdb.set_trace()

    # def _record_target_infos(self):
    #     self.target_cubes = []
    #     self.target_cubes_info = {}
    #     for cube_name in [self.ckpt['tgt_obj_name']]:
    #         for item in self.jimu_visual_cubes:
    #             if cube_name == item.name:
    #                 self.target_cubes.append(item)
    #                 self.target_cubes_info[cube_name] = {}
    #                 self.target_cubes_info[cube_name]["obj_pos"] = self.sim.data.body_xpos[self.sim.model.body_name2id(item.root_body)]
    #                 self.target_cubes_info[cube_name]["obj_xmat"] = self.sim.data.body_xmat[self.sim.model.body_name2id(item.root_body)]
    #     assert len(self.target_cubes) == len(self.jimu_tgt_cubes), \
    #         f"num of jimu_tgt_cubes: {len(self.jimu_tgt_cubes)}, however, num of target_cubes: {len(self.target_cubes)}"
    
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            
            @sensor(modality=modality)
            def moved_correct_jimu_objects_type(obs_cache):
                types = []
                for item in self.moved_correct_jimu_types:
                    types.extend(item)
                return np.array(types)
            
            @sensor(modality=modality)
            def moved_correct_jimu_objects_pos(obs_cache):
                poses = []
                for item in self.moved_correct_jimu_objects:
                    poses.extend(self.sim.data.body_xpos[self.sim.model.body_name2id(item.root_body)])
                return np.array(poses)

            @sensor(modality=modality)
            def moved_correct_jimu_objects_quat(obs_cache):
                quats = []
                for item in self.moved_correct_jimu_objects:
                    quats.extend(convert_quat(np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(item.root_body)]), to="xyzw"))
                return np.array(quats)
            
            @sensor(modality=modality)
            def moved_incorrect_jimu_objects_type(obs_cache):
                types = []
                for item in self.moved_incorrect_jimu_types:
                    types.extend(item)
                return np.array(types)
            
            @sensor(modality=modality)
            def moved_incorrect_jimu_objects_pos(obs_cache):
                poses = []
                for item in self.moved_incorrect_jimu_objects:
                    poses.extend(self.sim.data.body_xpos[self.sim.model.body_name2id(item.root_body)])
                return np.array(poses)

            @sensor(modality=modality)
            def moved_incorrect_jimu_objects_quat(obs_cache):
                quats = []
                for item in self.moved_incorrect_jimu_objects:
                    quats.extend(convert_quat(np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(item.root_body)]), to="xyzw"))
                return np.array(quats)

            @sensor(modality=modality)
            def virtual_jimu_objects_type(obs_cache):
                types = []
                for item in self.moved_correct_jimu_types:
                    types.extend(item)
                return np.array(types)
            
            @sensor(modality=modality)
            def virtual_jimu_objects_pos(obs_cache):
                poses = []
                for item in self.virtual_jimu_objects:
                    poses.extend(self.sim.data.body_xpos[self.sim.model.body_name2id(item.root_body)])
                return np.array(poses)
            
            @sensor(modality=modality)
            def virtual_jimu_objects_quat(obs_cache):
                poses = []
                for item in self.virtual_jimu_objects:
                    poses.extend(convert_quat(np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(item.root_body)]), to="xyzw"))
                return np.array(poses)

            @sensor(modality=modality)
            def gripper_to_moved_correct_jimu_objects(obs_cache):
                poses = []
                if "moved_correct_jimu_objects_pos" in obs_cache and f"{pf}eef_pos" in obs_cache:
                    for start_idx in range(self.num_correct_jimu_objects):
                        poses.extend(obs_cache["moved_correct_jimu_objects_pos"][start_idx * 3: (start_idx + 1) * 3] - obs_cache[f"{pf}eef_pos"])
                
                else:
                    for start_idx in range(self.num_correct_jimu_objects):
                        poses.extend(np.zeros(3))
                        
                return np.array(poses)

            @sensor(modality=modality)
            def gripper_to_virtual_jimu_objects(obs_cache):
                poses = []
                if "virtual_jimu_objects_pos" in obs_cache and f"{pf}eef_pos" in obs_cache:
                    for start_idx in range(self.num_correct_jimu_objects):
                        poses.extend(obs_cache["virtual_jimu_objects_pos"][start_idx * 3: (start_idx + 1) * 3] - obs_cache[f"{pf}eef_pos"])
                
                else:
                    for start_idx in range(self.num_correct_jimu_objects):
                        poses.extend(np.zeros(3))
                        
                return np.array(poses)
                

            @sensor(modality=modality)
            def moved_correct_to_virtual_pos(obs_cache):
                poses = []
                if "moved_correct_jimu_objects_pos" in obs_cache and "virtual_jimu_objects_pos" in obs_cache:
                    for start_idx in range(self.num_correct_jimu_objects):
                        poses.extend(
                            obs_cache["moved_correct_jimu_objects_pos"][start_idx * 3: (start_idx + 1) * 3] - \
                                obs_cache["virtual_jimu_objects_pos"][start_idx * 3: (start_idx + 1) * 3]
                        )
                else:
                    for start_idx in range(self.num_correct_jimu_objects):
                        poses.extend(np.zeros(3))
                        
                return np.array(poses)
            
            @sensor(modality=modality)
            def moved_correct_to_virtual_rotation(obs_cache):
                rotations = []
                if "moved_correct_jimu_objects_quat" in obs_cache and "virtual_jimu_objects_quat" in obs_cache:
                    for start_idx in range(self.num_correct_jimu_objects):
                        rotations.extend(
                            [abs(math.cos((quat2axisangle(obs_cache["moved_correct_jimu_objects_quat"][start_idx * 4: (start_idx + 1) * 4])[-1] - \
                                quat2axisangle(obs_cache["virtual_jimu_objects_quat"][start_idx * 4: (start_idx + 1) * 4])[-1]) % math.pi))]
                        )
                else:
                    for start_idx in range(self.num_correct_jimu_objects):
                        # rotations.extend(np.zeros(3))
                        rotations.extend([0])
                        
                return np.array(rotations)

            # sensors = [cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat, gripper_to_cubeA, gripper_to_cubeB, cubeA_to_cubeB]
            sensors = [
                moved_correct_jimu_objects_type,
                moved_correct_jimu_objects_pos,
                moved_correct_jimu_objects_quat,
                moved_incorrect_jimu_objects_type,
                moved_incorrect_jimu_objects_pos,
                moved_incorrect_jimu_objects_quat,
                virtual_jimu_objects_type,
                virtual_jimu_objects_pos,
                virtual_jimu_objects_quat,
                gripper_to_moved_correct_jimu_objects,
                gripper_to_virtual_jimu_objects,
                moved_correct_to_virtual_pos,
                moved_correct_to_virtual_rotation
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        # pdb.set_trace()
        return observables

    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        _, _, r_stack = self.staged_rewards()
        return r_stack > 0

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.moved_correct_jimu_objects[-1])