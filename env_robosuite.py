import gymnasium.spaces.box
import robosuite as suite
from robosuite.wrappers import GymWrapper

from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
import json
import pdb
from tianshou.env import DummyVectorEnv
import numpy as np
import reward_utils
import gymnasium

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

robosuite_tasks = ["lift", "stack", "jimu"]
from utils import print_green

class MyGymWrapper(GymWrapper):
    
    def _flatten_obs(self, obs_dict, verbose=False):
        data = super()._flatten_obs(obs_dict, verbose)
        self.obs_dict = obs_dict
        return data
    
    def __init__(self, env, keys=None, check_success=False):
        super().__init__(env, keys)
        self.check_success_flag = check_success
    ####### Using the Comments for reset when success #######
    
    def reset(self, seed=None, options=None):
        self.check_success_lasts = []
        return super().reset(seed, options)

    def step(self, action):
        print_green(f"action: {action}")
        o, r, terminated_R, truncated_R, info = super().step(action)
        if self.check_success_flag:
            if not terminated_R:
                # not reach max horizon (for example, 500 steps)
                if r >= 0.99:
                    self.check_success_lasts.append(True)
                else:
                    self.check_success_lasts.append(False)
                    
                terminated_R = all(self.check_success_lasts[-5:])
                if terminated_R:
                    print(GREEN + "====================Success====================" + RESET)
            
        if terminated_R and r >= 0.99:
            r = r
        else:
            r = 0.0
        return o, r, terminated_R, truncated_R, info
    
    ####### Using the Comments for reset when success #######
    
class ClipActionMyGymWrapper(MyGymWrapper):
    
    def __init__(self, env, keys=None, check_success=False):
        super().__init__(env, keys, check_success)
        cliped_action_dim = 3
        self.cliped_action_dim = cliped_action_dim
        self.action_space = gymnasium.spaces.box.Box(low=-1., high=1., shape=(self.action_dim - cliped_action_dim, ))
        # seems no useful
        # self.action_dim = self.action_dim - cliped_action_dim
        # setattr(self, "action_dim", self.action_dim - cliped_action_dim)
        # self.action_spec = (self.action_space.low, self.action_space.high)
        # pdb.set_trace()
        
    def step(self, action):
        act_action = np.zeros(self.action_dim)
        act_action[:3] = action[:3]
        act_action[-1] = action[-1]
        # action[3:6] = 0
        return super().step(act_action)

class OfflineMyGymWrapper(MyGymWrapper):
    def step(self, action):
        action[3:6] = 0
        return super().step(action)
    
def fetch_obs(env, obs):
    return obs
    # custom_envs = ["Jimu"]
    # if ENV_NAME not in custom_envs:
    #     return obs
    # # original observation info
    # eef_pos = obs['robot0_eef_pos']
    # eef_quat = obs['robot0_eef_quat']
    # gripper_qpos = obs['robot0_gripper_qpos']

    # # get cube position
    # achieved_goal, desired_goal = env.get_cube_pos()

    # return np.r_[eef_pos, eef_quat, gripper_qpos,
    #              achieved_goal, desired_goal]
    
def rotation_matrix(rot_angle, axis = 'z'):
    if axis == "x":
        return TU.quat2mat(np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0]))
    elif axis == "y":
        return TU.quat2mat(np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0]))
    elif axis == "z":
        return TU.quat2mat(np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]))
    
from robosuite.utils import transform_utils as TU

class LiftedMyGymWrapper(MyGymWrapper):
    
    def reset(self, seed=None, options=None):        
        while True:
            obs, info = super().reset(seed, options)
            
            curr_state = fetch_obs(self, obs)
            action_ori=TU.quat2axisangle(TU.mat2quat(self.robots[0].controller.ee_ori_mat))#TU.quat2axisangle(rotation_world)
            
            args_num = 20
            reaching_steps = args_num
            picking_steps = args_num
            gripper_steps = args_num // 2  # twice
            lifting_steps = args_num // 2  # twice
            moving_steps = args_num
            placing_steps = args_num
            
            pick_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(self.cubeA.root_body)]
            place_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(self.cubeB.root_body)]

            final_angle = TU.quat2axisangle(TU.mat2quat(rotation_matrix(0.5*np.pi, axis="x")@rotation_matrix(0, axis="y")@rotation_matrix(0, axis='z')))


            # 1.reach
            # reaching
            action = np.zeros(7)
            action[:3] = pick_pos
            action[3:6] = final_angle
            action[2] += 0.1
            for i in range(reaching_steps):
                obs, reward, done, _, info = self.step(action)  # take action in the environment
                next_state = fetch_obs(self, obs)
                curr_state = next_state

            # print("reaching reward", reward, "check_success", self._check_success())

            # 2.picking
            action[:3] = pick_pos
            action[3:6] = final_angle
            for i in range(picking_steps):
                obs, reward, done, _, info = self.step(action)  # take action in the environment
                next_state = fetch_obs(self, obs)
                curr_state = next_state

            # print("picking reward", reward, "check_success", self._check_success())
        
            # 3.picking gripper
            action[6] = 1
            for i in range(gripper_steps):
                obs, reward, done, _, info = self.step(action)  # take action in the environment
                next_state = fetch_obs(self, obs)
                curr_state = next_state

            # print("picking gripper reward", reward, "check_success", self._check_success())

            # 4.lifting
            action[2] += 0.2
            for i in range(lifting_steps):
                obs, reward, done, _, info = self.step(action)  # take action in the environment
                next_state = fetch_obs(self, obs)
                curr_state = next_state

            # print("lifting reward", reward, "check_success", self._check_success())
            
            # pdb.set_trace()
            if reward >= 0.5:
                break
        
        return obs, {}

class VideoMyGymWrapper(MyGymWrapper):
    def __init__(self, env, keys=None, check_success=False):
        super().__init__(env, keys, check_success)
        self.image_data = []
        
    def _get_img(self):
        full_obs = self._get_observations()
        img = full_obs[self.camera_names[0] + "_image"]
        return img
        
    def reset(self, seed=None, options=None):
        data = super().reset(seed, options)
        self.image_data.append(self._get_img())
        return data
    
    def step(self, action):
        data = super().step(action)
        self.image_data.append(self._get_img())
        return data
    

class VideoClipActionMyGymWrapper(ClipActionMyGymWrapper):
    def __init__(self, env, keys=None, check_success=False):
        super().__init__(env, keys, check_success)
        self.image_data = []
        
    def _get_img(self):
        full_obs = self._get_observations()
        img = full_obs[self.camera_names[0] + "_image"]
        return img
        
    def reset(self, seed=None, options=None):
        data = super().reset(seed, options)
        self.image_data.append(self._get_img())
        return data
    
    def step(self, action):
        data = super().step(action)
        self.image_data.append(self._get_img())
        return data
        
        

def init_envs(
    variant, 
    training_num, 
    test_num, 
    video=False, 
    control_freq=None, 
    meta_world=False, 
    lifted=False,
    demo_gen=False,
    obs_norm=False,
    control_delta=True,
    uncouple_pos_ori=True,
    control_ori=False,
    clipped_env=False,
    check_success=False,
    bc=False,
):
    if video:
        if clipped_env:
            Wrapper = VideoClipActionMyGymWrapper
        else:
            Wrapper = VideoMyGymWrapper
    else:
        if meta_world:
            Wrapper = StackMetaRewardWrapper
        else:
            if lifted:
                Wrapper = LiftedMyGymWrapper
            else:
                if clipped_env:
                    # Wrapper = OfflineMyGymWrapper
                    Wrapper = ClipActionMyGymWrapper
                else:
                    Wrapper = MyGymWrapper
    suites = []
    for env_config, number in zip((variant["expl_environment_kwargs"], variant["eval_environment_kwargs"]), (training_num, test_num)):
        
        if bc:
            env_config["reward_shaping"] = False
        
        if control_freq is not None:
            env_config["control_freq"] = control_freq
        
        # Load controller
        controller = env_config.pop("controller")
        if controller in set(ALL_CONTROLLERS):
            # This is a default controller
            controller_config = load_controller_config(default_controller=controller)
        else:
            # This is a string to the custom controller
            controller_config = load_controller_config(custom_fpath=controller)
                
        controller_config['control_delta'] = control_delta
        controller_config['uncouple_pos_ori'] = uncouple_pos_ori
        controller_config['control_ori'] = control_ori
            
        # if bc:
        #     keys = [
        #         'object-state', # 'object-state'
        #         'robot0_joint_pos_cos', 
        #         'robot0_joint_pos_sin',
        #         # "robot0_joint_pos", 
        #         # 'robot0_joint_vel', 
        #         'robot0_eef_pos', 
        #         'robot0_eef_quat', 
        #         'robot0_gripper_qpos', 
        #         # 'robot0_gripper_qvel'
        #     ]
        # else:
        # keys = ["object-state", "robot0_proprio-state"] #UR5e
        keys = [
            "object-state",
            'robot0_joint_pos_cos', 
            'robot0_joint_pos_sin',
            # "robot0_joint_pos", 
            'robot0_joint_vel', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos', 
            'robot0_gripper_qvel'
        ]
        suites.append(DummyVectorEnv([lambda: Wrapper(suite.make(**env_config,
                                 has_offscreen_renderer=video,
                                 use_camera_obs=video,
                                 controller_configs=controller_config,
                                 ), keys, check_success) for _ in range(number)]))
        
    if demo_gen:
        env = suite.make(
            **env_config,
            has_offscreen_renderer=video,
            use_camera_obs=video,
            controller_configs=controller_config,
        )
    else:
        env = Wrapper(
            suite.make(
                **env_config,
                has_offscreen_renderer=video,
                use_camera_obs=video,
                controller_configs=controller_config,
            ), keys, check_success
        )
    
    train_envs = suites[0]
    test_envs = suites[1]
    # pdb.set_trace()
    if obs_norm:
        from tianshou.env import BaseVectorEnv, VectorEnvNormObs
        train_envs = VectorEnvNormObs(train_envs, update_obs_rms=True)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return train_envs, test_envs, env

from gymnasium import spaces

class RGBDGymWrapper(GymWrapper):
    def __init__(self, env, keys=None):
        super().__init__(env, keys)
        
        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.shape
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)
        
    
    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]))
        
        return np.transpose(np.concatenate(ob_lst, axis=-1), (2, 0, 1)) # (h, w, d) --> (d, h, w)
    
class MyRGBDGymWrapper(RGBDGymWrapper):
    def __init__(self, env, keys=None):
        super().__init__(env, keys)
        self.ep_cur_length = 0
        self.ep_max_length = 500
        
    def reset(self, seed=None, options=None):
        self.ep_cur_length = 0
        return super().reset(seed, options)
    
    def step(self, action):
        o, r, terminated_R, truncated_R, info = super().step(action)
        if self._check_success():
            terminated_R = True
        self.ep_cur_length += 1
        if self.ep_cur_length >= self.ep_max_length:
            terminated_R = True
            
        return o, r, terminated_R, truncated_R, info
    
    
def init_envs_rgbd(variant, training_num, test_num, video=False, control_freq=None):
    suites = []
    for env_config, number in zip((variant["expl_environment_kwargs"], variant["eval_environment_kwargs"]), (training_num, test_num)):
        if control_freq is not None:
            env_config["control_freq"] = control_freq
        
        # Load controller
        controller = env_config.pop("controller")
        if controller in set(ALL_CONTROLLERS):
            # This is a default controller
            controller_config = load_controller_config(default_controller=controller)
        else:
            # This is a string to the custom controller
            controller_config = load_controller_config(custom_fpath=controller)
            
            
        keys = ["agentview_image", "agentview_depth"] # UR5e image
        suites.append(DummyVectorEnv([lambda: MyRGBDGymWrapper(suite.make(**env_config,
                                 has_renderer=False,
                                 reward_shaping=True,
                                 controller_configs=controller_config,
                                 ), keys) for _ in range(number)]))
        
    env = MyRGBDGymWrapper(suite.make(**env_config,
                                 has_renderer=False,
                                 reward_shaping=True,
                                 controller_configs=controller_config,
                                 ), keys)
    
    train_envs = suites[0]
    test_envs = suites[1]
    # pdb.set_trace()
    return train_envs, test_envs, env
    
from copy import deepcopy

class StackMetaRewardWrapper(MyGymWrapper):
    
    def __init__(self, env, keys=None, check_success=False):
        super().__init__(env, keys, check_success)
        self._TARGET_RADIUS: float = 0.05
        
    def reset(self, seed=None, options=None):
        data = super().reset(seed, options)
        self.init_obj_pos, self.init_tar_pos = self.get_cubeAB_pos()
        self.cubeB2target = np.array([0, 0, self._TARGET_RADIUS])
        self.init_tar_pos += self.cubeB2target # target = cubeB + 0.1 in the z axis
        self.init_left_pad, self.init_right_pad = self.get_fingerpad_pos()
        self.init_tcp = self.tcp_center
        return data
    
    def step(self, action):
        o, r, terminated_R, truncated_R, info = super().step(action)
        if r < 0.99:
            r = self.meta_reward(action, None)
        else:
            # success
            r = 10
        return o, r, terminated_R, truncated_R, info
    # @property
    # def _target_pos(self):
    #     x, y, z = self.sim.data.body_xpos[self.sim.model.body_name2id(self.cubeB.root_body)]
    #     _TARGET_RADIUS: float = 0.05
    #     return np.array([x, y, z + _TARGET_RADIUS])
    
    @property
    def tcp_center(self):
        """The COM of the gripper's 2 fingers.

        Returns:
            3-element position.
        """
        # right_finger_pos = self.data.site("rightEndEffector")
        # left_finger_pos = self.data.site("leftEndEffector")
        # tcp_center = (right_finger_pos.xpos + left_finger_pos.xpos) / 2.0
        # return tcp_center
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        return deepcopy(gripper_site_pos)
    
    def get_cubeAB_pos(self):
        cubeA_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(self.cubeA.root_body)]
        cubeB_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(self.cubeB.root_body)]
        return deepcopy(cubeA_pos), deepcopy(cubeB_pos)
    
    def get_fingerpad_pos(self):
        # print(env.sim.model.geom_names)
        left_pad = self.sim.data.geom_xpos[self.sim.model.geom_name2id("gripper0_left_fingerpad_collision")]
        right_pad = self.sim.data.geom_xpos[self.sim.model.geom_name2id("gripper0_right_fingerpad_collision")]
        return deepcopy(left_pad), deepcopy(right_pad)
        
    def meta_reward(self, action, obs):
        # maybe need update _target_pos for each step
        # assert self._target_pos is not None and self.init_obj_pos is not None
        cubeA, cubeB = self.get_cubeAB_pos()
        # print(f"cubeA: {cubeA}, cubeB: {cubeB}, init_obj_obs: {self.init_obj_pos}")
        
        
        tcp = self.tcp_center
        obj = cubeA
        # tcp_opened = obs[3] ### how????????????????????????????????????
        # tcp_opened = action[-1] - 0.2 # the last of the action control the open/close of the gripper, when action[-1] = -1/1 denotes open/close
        tcp_opened = -1 * action[-1] # the last of the action control the open/close of the gripper, when action[-1] = -1/1 denotes open/close
        target = cubeB + self.cubeB2target

        obj_to_target = float(np.linalg.norm(obj - target))
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        in_place_margin = np.linalg.norm(self.init_obj_pos - target)

        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, self._TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )

        object_grasped = self._gripper_caging_reward(action, obj)
        in_place_and_object_grasped = reward_utils.hamacher_product(
            object_grasped, in_place
        )
        reward = in_place_and_object_grasped
        
        # print(f"tcp_to_obj: {tcp_to_obj}, tcp_opened: {tcp_opened}, delta_obj_z: {obj[2] - self.init_obj_pos[2]}")

        if (
            tcp_to_obj < 0.02
            and (tcp_opened > 0)
            and (obj[2] - 0.01 > self.init_obj_pos[2])
        ):  
            # pdb.set_trace()
            reward += 1.0 + 5.0 * in_place
        # if obj_to_target < _TARGET_RADIUS: # I have delta in target with cubeB
            # pdb.set_trace()
        # if self._check_success(): 
        # #leave the success in the step
        #     reward = 10.0
        # return (reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place)
        return reward
    
    
    def _gripper_caging_reward(
        self,
        action,
        obj_pos,
        obj_radius: float = 0,  # All of these args are unused, just here to match
        pad_success_thresh: float = 0,  # the parent's type signature
        object_reach_radius: float = 0,
        xz_thresh: float = 0,
        desired_gripper_effort: float = 1.0,
        high_density: bool = False,
        medium_density: bool = False,
    ) -> float:
        pad_success_margin = 0.05
        x_z_success_margin = 0.005 # 0.005
        obj_radius = 0.015
        tcp = self.tcp_center
        # left_pad = self.get_body_com("leftpad") # Difference between leftEndEffector and leftpad in Meta-World?
        # right_pad = self.get_body_com("rightpad")
        left_pad, right_pad = self.get_fingerpad_pos()
        delta_object_y_left_pad = left_pad[1] - obj_pos[1]
        delta_object_y_right_pad = obj_pos[1] - right_pad[1]
        right_caging_margin = abs(
            abs(obj_pos[1] - self.init_right_pad[1]) - pad_success_margin
        )
        left_caging_margin = abs(
            abs(obj_pos[1] - self.init_left_pad[1]) - pad_success_margin
        )

        right_caging = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        y_caging = reward_utils.hamacher_product(left_caging, right_caging)

        # compute the tcp_obj distance in the x_z plane
        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
        obj_position_x_z = np.copy(obj_pos) + np.array([0.0, -obj_pos[1], 0.0])
        tcp_obj_norm_x_z = float(np.linalg.norm(tcp_xz - obj_position_x_z, ord=2))

        # used for computing the tcp to object object margin in the x_z plane
        assert self.init_obj_pos is not None
        init_obj_x_z = self.init_obj_pos + np.array([0.0, -self.init_obj_pos[1], 0.0])
        init_tcp_x_z = self.init_tcp + np.array([0.0, -self.init_tcp[1], 0.0])
        tcp_obj_x_z_margin = (
            np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
        )

        x_z_caging = reward_utils.tolerance(
            tcp_obj_norm_x_z,
            bounds=(0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )

        gripper_closed = min(max(0, action[-1]), 1)
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)

        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)
        caging_and_gripping = (caging_and_gripping + caging) / 2
        return caging_and_gripping
    

if __name__ == "__main__": 
    # test the sim speed of robosuite
    from OpenGL import GL
    def ignore_gl_errors(*args, **kwargs):
        pass
    GL.glCheckError = ignore_gl_errors
    import time, datetime, imageio
    import gymnasium as gym
    choose_robosuite = True
    video = True
    save_video_flag = True
    lifted = True
    control_freq = 20
    from eval_robosuite import save_video
    
    if choose_robosuite:
        task = "sunmao_jimu"
        variant = json.load(open(f"./config/{task}.json"))
        # init_envs(variant, 1, 1, False)
        env_config = variant["expl_environment_kwargs"]
        if control_freq is not None:
            env_config["control_freq"] = control_freq
        # Load controller
        controller = env_config.pop("controller")
        if controller in set(ALL_CONTROLLERS):
            # This is a default controller
            controller_config = load_controller_config(default_controller=controller)
        else:
            # This is a string to the custom controller
            controller_config = load_controller_config(custom_fpath=controller)
        controller_config['control_delta']=True # True
        controller_config['uncouple_pos_ori']=True
        # if video:
        #     env_config["render_camera"] = "frontview"
        #     env_config["camera_names"] = "frontview"
        #     env_config["camera_heights"] = 512
        #     env_config["camera_widths"] = 512
            
        # env_config["control_freq"] = 20
        env_suite = suite.make(
            **env_config,
            has_offscreen_renderer=video,
            use_camera_obs=video,
            controller_configs=controller_config,
        )
        # keys = ["object-state"]
        keys = []
        for idx in range(len(env_suite.robots)):
            keys.append(f"robot{idx}_proprio-state")
        # if video:
        #     env = VideoMyGymWrapper(env_suite, keys)
        # else:
        #     env = MyGymWrapper(env_suite, keys)
            # env = StackMetaRewardWrapper(env_suite, keys)
            # env = LiftedMyGymWrapper(env_suite, keys)
        # env = GymWrapper(env_suite, keys)
        env = env_suite
    else:
        env = gym.make("Pendulum-v1")
    
    
    # controller_config = load_controller_config(default_controller="OSC_POSE")
    # controller_config['control_delta']=True
    # controller_config['uncouple_pos_ori']=True

    # # create environment instance
    # env = suite.make(
    #     #env_name="NutAssembly", # try with other tasks like "Stack" and "Door"
    #     #env_name="PickPlace", # try with other tasks like "Stack" and "Door"
    #     #env_name="Mstt", # try with other tasks like "Stack" and "Door"
    #     env_name="Jimu", # try with other tasks like "Stack" and "Door"
    #     #env_name="Sunmao", # try with other tasks like "Stack" and "Door"
    #     #env_name="Stack", # try with other tasks like "Stack" and "Door"
    #     robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
    #     gripper_types = "default",
    #     controller_configs = controller_config,

    #     #has_renderer=True,
    #     #has_offscreen_renderer=False,
    #     #use_camera_obs = False,
    #     #render_camera = "frontview",
    #     #control_freq = 20,

    #     has_renderer=False,
    #     has_offscreen_renderer=True,
    #     render_gpu_device_id=0,
    #     horizon=500,
    #     render_camera="frontview",
    #     camera_names="frontview", 
    #     use_object_obs=False,
    #     use_camera_obs=True,
    #     control_freq=20,
    #     camera_depths=True,
    #     camera_heights=512,
    #     camera_widths=512,
    #     reward_shaping=True,
    # )
    eps = 10
    i_eps = 0
    step_count = 0
    start_time = time.time()
    images = []
    front_images = []
    agent_images = []
    robot_images = []
    while i_eps < eps:
        # pdb.set_trace()
        obs = env.reset()
        # pdb.set_trace()
        front_images.append(obs["frontview_image"])
        # agent_images.append(obs["agentview_image"])
        # robot_images.append(obs["robot0_eye_in_hand_image"])
        # pdb.set_trace()
        # images.append(obs["agentview_image"])
        # images.append(obs["robot0_eye_in_hand_image"])
        # pdb.set_trace()
        # env.action_space.seed(seed=1)
        d = False
        print(f"eps: {i_eps}, steps: {step_count}")
        ep_step_count = 0
        while not d:
            # pdb.set_trace()
            # print(f"eps: {i_eps}, steps: {step_count}")
            action = np.zeros(7)
            # action[:3] = 0.0
            # action[2] = 1.0
            # obs, r, d1, d2, _ = env.step(env.action_space.sample())
            # obs, r, d1, d2, _ = env.step(action)
            # d = d1 or d2
            obs, r, d, _ = env.step(action)
            env._check_success()
            front_images.append(obs["frontview_image"])
            # agent_images.append(obs["agentview_image"])
            # robot_images.append(obs["robot0_eye_in_hand_image"])
            # pdb.set_trace()
            # images.append(obs["agentview_image"])
            # images.append(obs["robot0_eye_in_hand_image"])
            # _, r, d1 , _ = env.step([0, 0, 0, 0, 0, 0, 0])
            # d = d1
            step_count += 1
            ep_step_count += 1
            if ep_step_count >= 50:
                break
            if not choose_robosuite:
                if ep_step_count >= 50:
                    break
                
            # print(f"reward: {r}")
            # print(f"eps: {i_eps}, steps: {step_count}, reward: {r}, d: {d}")
        i_eps += 1
        # save_video("./sunmao_jimu.mp4", [images])
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_step = total_time / step_count
    print(f"time_per_step: {time_per_step}")
    
    # save video 
    if video and save_video_flag:
        print(f"save video")
        # images.extend(front_images)
        images.extend(front_images)
        # images.extend(robot_images)
        # save_video("./jimu.mp4", [env.image_data])
        save_video("./sunmao_jimu.mp4", [images])