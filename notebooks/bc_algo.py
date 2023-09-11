import torch
import torch.nn as nn

from torchvision.models.detection import maskrcnn_resnet50_fpn

from libero.lifelong.algos import Sequential
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.language_modules import MLPEncoder
from libero.lifelong.models.modules.rgb_modules import ResnetEncoder
from libero.lifelong.models.policy_head import GMMHead

class ExtraModalities:
    def __init__(self,
                 use_joint=False,
                 use_gripper=False,
                 use_ee=False,
                 use_bounding_box=False,
                 extra_hidden_size=64,
                 extra_embedding_size=32):

        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size
        self.use_bounding_box = use_bounding_box

        # neo: added a bounding box encoder
        self.bounding_box_encoder = maskrcnn_resnet50_fpn(pretrained=True)

        joint_states_dim = 7
        gripper_states_dim = 2
        ee_dim = 6

        self.extra_low_level_feature_dim = int(use_joint) * joint_states_dim + \
                                           int(use_gripper) * gripper_states_dim + \
                                           int(use_ee) * ee_dim
        assert self.extra_low_level_feature_dim > 0, "[error] no extra information"

    def __call__(self, obs_dict):
        """
        obs_dict: {
            (optional) joint_stats: (B, T, 7),
            (optional) gripper_states: (B, T, 2),
            (optional) ee: (B, T, 3)
        }
        map above to a latent vector of shape (B, T, H)
        """
        tensor_list = []
        if self.use_joint:
            tensor_list.append(obs_dict["joint_states"])
        if self.use_gripper:
            tensor_list.append(obs_dict["gripper_states"])
        if self.use_ee:
            tensor_list.append(obs_dict["ee_states"])
        x = torch.cat(tensor_list, dim=-1)
        return x

    def output_shape(self, input_shape, shape_meta):
        return (self.extra_low_level_feature_dim,)


def get_encoder(encoder_name: str):
    if encoder_name == "ResnetEncoder":
        return ResnetEncoder
    return None


class BCPolicy(BasePolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self,
                 cfg,
                 shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        ### 1. encode image
        rnn_input_size = 0
        image_embed_size = 64
        self.image_encoders = {}

        # shape_meta["all_shapes"] is a dict of all the modalities
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = image_embed_size
                kwargs.language_dim = policy_cfg.language_encoder.network_kwargs.input_size
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": get_encoder(policy_cfg.image_encoder.network)(**kwargs)
                }
                rnn_input_size += image_embed_size
        self.encoders = nn.ModuleList([x["encoder"] for x in self.image_encoders.values()])

        ### 2. encode language
        text_embed_size = 32
        policy_cfg.language_encoder.network_kwargs.output_size = text_embed_size
        self.language_encoder = eval(policy_cfg.language_encoder.network)(
            **policy_cfg.language_encoder.network_kwargs)
        rnn_input_size += text_embed_size

        ### 3. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = ExtraModalities(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee)
        rnn_input_size += self.extra_encoder.extra_low_level_feature_dim
        bidirectional = False
        self.rnn = nn.LSTM(input_size=rnn_input_size,
                           hidden_size=1024,
                           num_layers=2,
                           batch_first=True,
                           dropout=0.0,
                           bidirectional=bidirectional)

        ### 4. use policy head to output action
        self.D = 2 if bidirectional else 1
        self.policy_head = GMMHead(
            input_size=1024,
            loss_coef=2.0,
            hidden_size=1024,
            num_layers=2,
            min_std=0.0001,
            num_modes=5,
            activation="softplus",
            output_size=shape_meta["ac_dim"])
        self.eval_h0 = None
        self.eval_c0 = None

    def forward(self, data, train_mode=True):
        # 1. encode image
        encoded = []
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            e = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"].reshape(B, 1, -1).repeat(1, T, 1).reshape(B * T, -1)
            ).view(B, T, -1)
            encoded.append(e)

        # 2. add joint states, gripper info, etc.
        encoded.append(self.extra_encoder(data["obs"]))  # add (B, T, H_extra)
        encoded = torch.cat(encoded, -1)  # (B, T, H_all)

        # 3. language encoding
        lang_h = self.language_encoder(data)  # (B, H)
        encoded = torch.cat([encoded,
                             lang_h.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1)

        # 4. apply temporal rnn
        if train_mode:
            h0 = torch.zeros(self.D * 2,
                             encoded.shape[0],
                             1024).to(self.device)
            c0 = torch.zeros(self.D * 2,
                             encoded.shape[0],
                             1024).to(self.device)
            output, (hn, cn) = self.rnn(encoded, (h0, c0))
        else:
            if self.eval_h0 is None:
                self.eval_h0 = torch.zeros(
                    self.D * 2,
                    encoded.shape[0],
                    1024).to(self.device)
                self.eval_c0 = torch.zeros(
                    self.D * 2,
                    encoded.shape[0],
                    1024).to(self.device)
            output, (h1, c1) = self.rnn(encoded, (self.eval_h0, self.eval_c0))
            self.eval_h0 = h1.detach()
            self.eval_c0 = c1.detach()

        dist = self.policy_head(output)
        return dist

    def get_action(self, data):
        self.eval()
        data = self.preprocess_input(data, train_mode=False)
        with torch.no_grad():
            dist = self.forward(data)
        action = dist.sample().detach().cpu()
        return action.view(action.shape[0], -1).numpy()

    def reset(self):
        self.eval_h0 = None
        self.eval_c0 = None


class BehavioralCloningLifelongAlgo(Sequential):
    def __init__(self,
                 n_tasks,
                 cfg):
        super().__init__(n_tasks=n_tasks, cfg=cfg)
        # define the learning policy
        self.policy = BCPolicy(cfg, cfg.shape_meta)

    def start_task(self, start_task):
        # what to do at the beginning of a new task
        super().start_task(start_task)

    def observe(self, data):
        # how the algorithm observes a data and returns a loss to be optimized
        loss = super().observe(data)
        return loss
