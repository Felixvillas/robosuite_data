from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, SBoxVisualObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler, UniformFixSampler
from robosuite.utils.transform_utils import convert_quat

class JimuLift(SingleArmEnv):
    """
    This class corresponds to the jimu task for a single robot arm.

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

        jimu_size = 3,
        jimu_move_num = 1,
        jimu_src_matrix = None,
        jimu_tgt_matrix = None,
    ):
        # set jimu params
        self.jimu_src_m = jimu_src_matrix
        self.jimu_tgt_m = jimu_tgt_matrix
        self.jimu_move_num = jimu_move_num
        self.jimu_size = jimu_size
        if jimu_src_matrix == None:
            self.jimu_shape = (np.random.randint(1, self.jimu_size+1),
                                np.random.randint(1, self.jimu_size+1),
                                np.random.randint(1, self.jimu_size+1),
                               )
            self.jimu_shape = (1,5,5)
        else:
            self.jimu_shape = jimu_src_matrix.shape()

        #    num = self.jimu_shape[0] * self.jimu_shape[1] * self.jimu_shape[2]
        #    src_num = np.random.randint(0, num-jimu_move_num+1)
        #    #src_num = 4
        #    tgt_num = src_num + self.jimu_move_num

        #    def find_jimu_pos(m):
        #        z = 0 # np.random.randint(0, m.shape[0])
        #        y = np.random.randint(0, m.shape[1])
        #        x = np.random.randint(0, m.shape[2])
        #        if (not m[z,y,x]) and (z == 0 or m[z-1, y, x]) and np.random.rand() > 0.5:
        #            #print("find",z,y,x, z==0, (z == 0 or m[z-1, y, x]))
        #            m[z,y,x] = 1
        #            return True
        #        return False

        #    self.jimu_src_m = np.zeros(self.jimu_shape, bool)
        #    while(self.jimu_src_m.sum() <= src_num):
        #        find_jimu_pos(self.jimu_src_m)
        #
        #    self.jimu_tgt_m = self.jimu_src_m.copy()
        #    while(self.jimu_tgt_m.sum() <= tgt_num):
        #        find_jimu_pos(self.jimu_tgt_m)

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
        print("JimuLift Env Init Success")

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
        # reward = 0.0
        # r_reach, r_lift, r_stack = self.staged_rewards()
        # if self.reward_shaping:
        #     reward = max(r_reach, r_lift, r_stack)
        # else:
        #     reward = 2.0 if r_stack > 0 else 0.0

        # if self.reward_scale is not None:
        #     reward *= self.reward_scale / 2.0

        # return reward

        ## Lift task reward
        reward = 0.0

        # completion reward
        cubeA_pos = self.sim.data.body_xpos[self.tgt_cube_ids[-1]]
        cubeA_height = cubeA_pos[2]
        table_height = self.table_offset[2]
        if cubeA_height > table_height + 0.04:
            reward = 2.25

        # use shaping reward
        elif self.reward_shaping:
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.jimu_tgt_cubes[-1]):
                reward += 0.25

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

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
        # reaching is successful when the gripper site is close to the center of the cube
        #TODO: for multiple target cubes
        cubeA_pos = self.sim.data.body_xpos[self.tgt_cube_ids[-1]]
        # print(f"reward_cubeA_pos = {cubeA_pos}")
        cubeB_pos_ranges = self.tgt_cube_poses[-1]
        # print(f"reward_cubeB_pos_ranges = {cubeB_pos_ranges}")
        cubeB_pos = [
            (cubeB_pos_ranges[0][0]+cubeB_pos_ranges[0][1])/2,
            (cubeB_pos_ranges[1][0]+cubeB_pos_ranges[1][1])/2,
            (cubeB_pos_ranges[2][0]+cubeB_pos_ranges[2][1])/2-0.02,
                     ]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25
        print(f"Apos = {cubeA_pos} || Bpos = {cubeB_pos}")
        print(f"cubeB_pos_range=\n{cubeB_pos_ranges}")
        # print(f"cubeA_pos[2]={cubeA_pos[2]} || cubeB_pos_ranges[2][0]={cubeB_pos_ranges[2][0]}")

        # grasping reward
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.jimu_tgt_cubes[-1])
        if grasping_cubeA:
            r_reach += 0.25

        # lifting is successful when the cube is above the table top by a margin
        cubeA_height = cubeA_pos[2]
        table_height = self.table_offset[2]
        cubeA_lifted = cubeA_height > table_height + 0.04
        r_lift = 1.0 if cubeA_lifted else 0.0

        # Aligning is successful when cubeA is right above cubeB
        if cubeA_lifted:
            horiz_dist = np.linalg.norm(np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2]))
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_stack = 0
        #cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        #if not grasping_cubeA and r_lift > 0 and cubeA_touching_cubeB:
        #    r_stack = 2.0
        try:
            cubeA_in_right_pos = (cubeA_pos[0] >= cubeB_pos_ranges[0][0] and cubeA_pos[0] <= cubeB_pos_ranges[0][1]) and \
                                 (cubeA_pos[1] >= cubeB_pos_ranges[1][0] and cubeA_pos[1] <= cubeB_pos_ranges[1][1]) and \
                                    (cubeA_pos[2] >= cubeB_pos_ranges[2][0] and cubeA_pos[2] <= cubeB_pos_ranges[2][1])
            if not grasping_cubeA and r_lift > 0 and cubeA_in_right_pos:
                r_stack = 2.0
        except:
            breakpoint()

        return r_reach, r_lift, r_stack

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

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

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )


        max_size = max(self.jimu_shape)
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.jimu_src_cubes = []
        self.jimu_tgt_cubes = []
        self.jimu_visual_cubes = []
        for z in range(self.jimu_shape[0]):
            for y in range(self.jimu_shape[1]):
                for x in range(self.jimu_shape[2]):
                    y_range = [0.05*y + 0.1, 0.05*(y+1) + 0.1]
                    x_range = [0.05*x-0.025*max_size, 0.05*(x+1)-0.025*max_size]

                    obj_idx_name = str(z)+"_"+str(y)+"_"+str(x)
                    self.jimu_src_cubes.append(
                        BoxObject(
                            name="src_cube_"+ obj_idx_name,
                            size_min=[0.02, 0.02, 0.02],
                            size_max=[0.02, 0.02, 0.02],
                            rgba=[0, 1, 0, 1],
                            material=greenwood,
                        )
                    )


                    self.placement_initializer.append_sampler(UniformFixSampler(
                            name="SrcObjectSampler_"+obj_idx_name,
                            mujoco_objects=self.jimu_src_cubes[-1],
                            x_range=x_range,
                            y_range=y_range,
                            rotation=0,
                            rotation_axis = "x",
                            ensure_object_boundary_in_range=False,
                            ensure_valid_placement=False,
                            reference_pos=self.table_offset,
                            #z_offset=0.04*z+0.015,
                            z_offset=0.01,
                        )
                    )

                    visual_cube_name = "visual_cube_"+str(z)+"_"+str(y)+"_"+str(x)
                    self.jimu_visual_cubes.append(
                        SBoxVisualObject(visual_cube_name)
                    )

                    self.placement_initializer.append_sampler(UniformFixSampler(
                            name="VisualObjectSampler_"+visual_cube_name,
                            mujoco_objects=self.jimu_visual_cubes[-1],
                            x_range=x_range,
                            y_range=y_range,
                            rotation=0,
                            rotation_axis = "x",
                            ensure_object_boundary_in_range=False,
                            ensure_valid_placement=False,
                            reference_pos=self.table_offset,
                            #z_offset=0.04*z+0.015,
                            z_offset=0.01,
                        )
                    )
        #self.tgt_cube_poses = []
        #for z in range(self.jimu_tgt_m.shape[0]):
        #    for y in range(self.jimu_tgt_m.shape[1]):
        #        for x in range(self.jimu_tgt_m.shape[2]):
        #            if not self.jimu_src_m[z,y,x] and self.jimu_tgt_m[z,y,x]:
                        #y_range = [0.05*y + 0.1, 0.05*(y+1) + 0.1]
                        #x_range = [0.05*x-0.025*max_size, 0.05*(x+1)-0.025*max_size]
                        #z_range = [self.table_offset+0.05, self.table_offset+0.015]
                        #self.tgt_cube_poses.append([x_range, y_range, z_range])
                        #obj_idx_name = str(z)+"_"+str(y)+"_"+str(x)
        self.jimu_tgt_cubes.append(
            BoxObject(
                #name="tgt_cube_"+obj_idx_name,
                name="tgt_cube",
                size_min=[0.02, 0.02, 0.02],
                size_max=[0.02, 0.02, 0.02],
                rgba=[1, 0, 0, 1],
                material=redwood,
            )
        )

        self.placement_initializer.append_sampler(UniformFixSampler(
                name="TgtObjectSampler_tgt_cube",
                mujoco_objects=self.jimu_tgt_cubes[-1],
                x_range=[-0.04, 0.04],
                y_range=[-0.08, -0.00],
                rotation=0,
                rotation_axis = "x",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.jimu_src_cubes + self.jimu_tgt_cubes+self.jimu_visual_cubes,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.tgt_cube_ids = []
        for cube in self.jimu_tgt_cubes:
            self.tgt_cube_ids.append(self.sim.model.body_name2id(cube.root_body))
        # TODO
        if len(self.tgt_cube_ids) > 1:
            raise NotImplementedError


        self.obj_body_id = {}
        self.obj_geom_id = {}
        for obj in self.jimu_src_cubes + self.jimu_tgt_cubes+self.jimu_visual_cubes:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        ## low-level object information
        #if self.use_object_obs:
        #    # Get robot prefix and define observables modality
        #    pf = self.robots[0].robot_model.naming_prefix
        #    modality = "object"

        #    # cube-related observables
        #    @sensor(modality=modality)
        #    def cube_pos(obs_cache):
        #        return np.array(self.sim.data.body_xpos[self.cube_body_id])

        #    @sensor(modality=modality)
        #    def cube_quat(obs_cache):
        #        return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

        #    @sensor(modality=modality)
        #    def gripper_to_cube_pos(obs_cache):
        #        return (
        #            obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"]
        #            if f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache
        #            else np.zeros(3)
        #        )

        #    sensors = [cube_pos, cube_quat, gripper_to_cube_pos]
        #    names = [s.__name__ for s in sensors]

        #    # Create observables
        #    for name, s in zip(names, sensors):
        #        observables[name] = Observable(
        #            name=name,
        #            sensor=s,
        #            sampling_rate=self.control_freq,
        #        )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # sample distribution of jimu
        total_num = self.jimu_shape[0] * self.jimu_shape[1] * self.jimu_shape[2]
        src_num = np.random.randint(0, total_num-self.jimu_move_num+1)
        #src_num = 4
        tgt_num = src_num + self.jimu_move_num

        def find_jimu_pos(m):
            z = 0 # np.random.randint(0, m.shape[0])
            y = np.random.randint(0, m.shape[1])
            x = np.random.randint(0, m.shape[2])
            if (not m[z,y,x]) and (z == 0 or m[z-1, y, x]) and np.random.rand() > 0.5:
                #print("find",z,y,x, z==0, (z == 0 or m[z-1, y, x]))
                m[z,y,x] = 1
                return True, (z,y,x)
            return False, None

        src_obj_names = []
        self.jimu_src_m = np.zeros(self.jimu_shape, bool)
        while(self.jimu_src_m.sum() <= src_num):
            suc, zyx = find_jimu_pos(self.jimu_src_m)
            if suc:
                z, y, x = zyx
                obj_name = 'src_cube_' + str(z)+"_"+str(y)+"_"+str(x)
                src_obj_names.append(obj_name)
            if self.jimu_src_m.sum() == src_num: break

        tgt_obj_name = None
        self.jimu_tgt_m = self.jimu_src_m.copy()
        while(self.jimu_tgt_m.sum() <= tgt_num):
            suc, zyx = find_jimu_pos(self.jimu_tgt_m)
            if suc:
                 z, y, x = zyx
                 max_size = max(self.jimu_shape)
                 y_range = [0.05*y + 0.1, 0.05*(y+1) + 0.1]
                 x_range = [0.05*x-0.025*max_size, 0.05*(x+1)-0.025*max_size]
                 z_range = [self.table_offset[2]+0.05, self.table_offset[2]+0.015]
                 self.tgt_cube_poses = [[x_range, y_range, z_range]]
                 tgt_obj_name = "visual_cube_"+str(z)+"_"+str(y)+"_"+str(x)
            if self.jimu_tgt_m.sum() == tgt_num: break

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.name not in src_obj_names + [tgt_obj_name, 'tgt_cube']:
                    continue

                if "visual" in obj.name.lower():
                   self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                   self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                else:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

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
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        #cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        #table_height = self.model.mujoco_arena.table_offset[2]

        ## cube is higher than the table top above a margin
        #return cube_height > table_height + 0.04
        _, _, r_stack = self.staged_rewards()
        return r_stack > 0
