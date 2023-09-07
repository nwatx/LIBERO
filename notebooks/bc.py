import os
import hydra
import yaml
from easydict import EasyDict
from omegaconf import OmegaConf
from robomimic.utils.file_utils import get_shape_metadata_from_dataset

from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from bc_algo import BehavioralCloningLifelongAlgo
from libero.lifelong.datasets import get_dataset
from libero.lifelong.utils import create_experiment_dir
import robomimic.utils.obs_utils as ObsUtils


@hydra.main(config_path="../libero/configs", config_name="config", version_base=None)
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    cfg.folder = cfg.folder or get_libero_path("datasets")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_object"
    task_suite = benchmark_dict[task_suite_name]()

    # get the dataset metadata

    task_id = 0

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})

    # code to get shape metadata
    task_benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    dataset_path = os.path.join(cfg.folder, task_benchmark.get_task_demonstration(task_id))
    print(dataset_path)
    cfg.shape_meta = get_shape_metadata_from_dataset(dataset_path)

    # code to get the task
    task = task_suite.get_task(task_id)
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
          f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(
        task_id)  # for benchmarking purpose, we fix the a set of initial states
    init_state_id = 0
    obs = env.set_init_state(init_states[init_state_id])

    algo = BehavioralCloningLifelongAlgo(n_tasks=1, cfg=cfg)

    for step in range(10):
        # observe the current state
        algo.observe(obs)
        # predict the action
        action = algo.predict()
        # execute the action
        obs, reward, done, info = env.step(action)


if __name__ == "__main__":
    main()
