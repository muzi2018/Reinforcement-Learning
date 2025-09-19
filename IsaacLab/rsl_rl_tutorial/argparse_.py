import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Demo script using argparse")

# Add arguments
parser.add_argument("name", help="Your name")             # positional argument
parser.add_argument("-a", "--age", type=int, help="Your age")  # optional argument
# parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed output")

# Parse the arguments
# args = parser.parse_args()
# # KEYPOINT: parse_known_args returns a tuple: (known_args, unknown_args)
args, unknown = parser.parse_known_args()

print("Known args:", args)
print("Unknown args:", unknown)

# Use arguments
print(f"Hello, {args.name}!")
if args.age:
    print(f"You are {args.age} years old.")
if args.verbose:
    print("Verbose mode is on.")


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg



""""
rsl_rl/train.py
 general arg: --video --video_length --video_interval --num_envs --task --seed --max_iterations --distributed 
 rsl_rl arg: --experiment_name --run_name --resume --load_run --checkpoint --logger --log_project_name
"""