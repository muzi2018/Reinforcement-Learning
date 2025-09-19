import functools
from typing import Callable

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    raise ImportError("Hydra is not installed. Run 'pip install hydra-core'")

# Dummy config classes
class DummyEnvCfg:
    def __init__(self):
        self.param = 0
        self.num_envs = 1
        self.seed = 0
    def from_dict(self, d):
        self.param = d.get("param", self.param)
        self.num_envs = d.get("num_envs", self.num_envs)
        self.seed = d.get("seed", self.seed)
    def __repr__(self):
        return f"DummyEnvCfg(param={self.param}, num_envs={self.num_envs}, seed={self.seed})"

class DummyAgentCfg:
    def __init__(self):
        self.lr = 0.01
        self.max_iterations = 100
        self.clip_actions = True
        self.run_name = "default"
        self.seed = 0
        self.device = "cpu"
    def from_dict(self, d):
        self.lr = d.get("lr", self.lr)
        self.max_iterations = d.get("max_iterations", self.max_iterations)
        self.clip_actions = d.get("clip_actions", self.clip_actions)
        self.run_name = d.get("run_name", self.run_name)
        self.seed = d.get("seed", self.seed)
        self.device = d.get("device", self.device)
    def __repr__(self):
        return (
            f"DummyAgentCfg(lr={self.lr}, max_iterations={self.max_iterations}, "
            f"clip_actions={self.clip_actions}, run_name='{self.run_name}', seed={self.seed}, device='{self.device}')"
        )

# Register task to Hydra using actual YAML files
def register_task_to_hydra(task_name: str, agent_cfg_entry_point: str):
    # This tells Hydra to load configs from "configs/<task_name>/<yaml file>"
    # Here agent_cfg_entry_point is the name of the YAML file for the agent
    env_cfg = DummyEnvCfg()
    agent_cfg = DummyAgentCfg()

    # Store a node with placeholders, Hydra will override from YAML
    cfg_dict = {
        "env": env_cfg.__dict__,
        "agent": agent_cfg.__dict__
    }
    ConfigStore.instance().store(name=task_name, node=cfg_dict)
    return env_cfg, agent_cfg

# Hydra decorator
def hydra_task_config(task_name: str, agent_cfg_entry_point: str) -> Callable:
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            env_cfg, agent_cfg = register_task_to_hydra(task_name, agent_cfg_entry_point)

            @hydra.main(config_path="configs", config_name=task_name, version_base="1.3")
            def hydra_main(hydra_cfg: DictConfig):
                # load env config from YAML
                env_cfg.from_dict(hydra_cfg["env"])
                # load agent config from YAML
                agent_cfg.from_dict(hydra_cfg["agent"])
                func(env_cfg, agent_cfg, *args, **kwargs)

            hydra_main()
        return wrapper
    return decorator

# Example function
@hydra_task_config(task_name="my_dummy_task", agent_cfg_entry_point="dummy_agent")
def run_task(env_cfg, agent_cfg):
    print("Environment Config:", env_cfg)
    print("Agent Config:", agent_cfg)

if __name__ == "__main__":
    run_task()



# update rsl_rl cfg from cli args
"""
agent_cfg :
    update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace)
    --seed, --resume, --load_run, --checkpoint, --run_name, --logger, --wandb_project, --neptune_project
    --max_iterations
env_cfg : scene.num_envs, seed, sim.device, 
log_root_path :  os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
"""



