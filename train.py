import argparse
import logging
import os
import sys
import time
from datetime import datetime

import torch
from omegaconf import OmegaConf

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

LOCAL_IMPORT_PATHS = [
    os.path.join(ROOT_DIR, "assetto_corsa_gym"),
    os.path.join(ROOT_DIR, "algorithm", "discor"),
]
for path in reversed(LOCAL_IMPORT_PATHS):
    sys.path.insert(0, path)

import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from discor.agent import Agent
from discor.algorithm.sac import SAC
import Common.logging_config as logging_config
import Common.misc as misc

logger = logging.getLogger(__name__)


def wait_for_start() -> None:
    print("\n" + "=" * 50)
    print("Environment ready!")
    print("Launch Assetto Corsa, then press SPACE to start training...")
    print("=" * 50 + "\n")

    if os.name == "nt":
        import msvcrt

        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b" ":
                    print("Starting training...\n")
                    time.sleep(0.2)
                    return
            time.sleep(0.05)

    print("Spacebar polling is unavailable on this platform; press Enter to start training.")
    input()
    print("Starting training...\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Assetto Corsa SAC trainer")
    parser.add_argument("--config", default="config.yml", type=str, help="Config path")
    parser.add_argument("--load_path", type=str, default=None, help="Optional model dir to load")
    parser.add_argument(
        "--load_weights_only",
        action="store_true",
        help="Load model weights but skip replay buffer (useful for corrupted/incompatible buffers)",
    )
    parser.add_argument("--test", action="store_true", help="Run eval instead of training")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="OmegaConf dotlist overrides")
    return parser.parse_args()


def build_work_dir(config) -> str:
    if config.work_dir:
        work_dir = os.path.abspath(config.work_dir)
    else:
        work_dir = os.path.join(ROOT_DIR, "outputs", datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3])
        work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    return work_dir + os.sep


def maybe_load_demonstrations(agent: Agent, env, config) -> bool:
    demo_config = getattr(config, "Demonstrations", None)
    if demo_config is None or not getattr(demo_config, "enabled", False):
        return False

    data_paths = []
    single_path = getattr(demo_config, "data_path", None)
    if single_path:
        data_paths.append(single_path)
    for path in getattr(demo_config, "data_paths", []) or []:
        data_paths.append(path)

    if not data_paths:
        raise ValueError("Demonstrations.enabled is true, but no Demonstrations.data_path or data_paths were provided")

    total_transitions = 0
    log_steer_ratios = getattr(demo_config, "log_steer_ratios", False)
    for data_path in data_paths:
        abs_data_path = os.path.abspath(data_path)
        total_transitions += agent.load_pre_train_data(abs_data_path, env, log_steer_ratios=log_steer_ratios)

    if total_transitions <= 0:
        raise ValueError("No demonstration transitions were loaded from the configured paths")

    pretrain_epochs = int(getattr(demo_config, "pretrain_epochs", 0))
    if pretrain_epochs > 0:
        agent.pre_train_epochs(pretrain_epochs, num_samples=total_transitions)
        return True

    legacy_pretrain_steps = int(getattr(demo_config, "pretrain_steps", 0))
    if legacy_pretrain_steps > 0:
        logger.warning(
            "Demonstrations.pretrain_steps is deprecated. "
            "Use Demonstrations.pretrain_epochs so each epoch covers the demonstration dataset once."
        )
        agent.pre_train(legacy_pretrain_steps)
        return True

    logger.info("Demonstrations were loaded, but pretrain_epochs=0 so demo pretraining was skipped.")
    return False


def main() -> None:
    args = parse_args()

    config = OmegaConf.load(args.config)
    if args.overrides:
        cli_conf = OmegaConf.from_dotlist(args.overrides)
        config = OmegaConf.merge(config, cli_conf)

    config.work_dir = build_work_dir(config)

    logging_config.create_logging(level=logging.DEBUG, file_name=config.work_dir + "train.log")
    logging.getLogger().setLevel(logging.INFO)

    misc.get_system_info()
    misc.get_git_commit_info()
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    env = assettoCorsa.make_ac_env(cfg=config, work_dir=config.work_dir)
    logger.info("Environment initialized")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    algo = SAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
        seed=config.seed,
        **OmegaConf.to_container(config.SAC),
    )

    agent = Agent(
        env=env,
        test_env=env,
        algo=algo,
        log_dir=config.work_dir,
        device=device,
        seed=config.seed,
        wandb_logger=None,
        **config.Agent,
    )

    if args.load_path:
        load_path = os.path.abspath(args.load_path)
        if not load_path.endswith(os.sep):
            load_path += os.sep
        agent.load(load_path, load_buffer=(not args.test and not args.load_weights_only))

    if not args.test:
        did_demo_pretrain = maybe_load_demonstrations(agent, env, config)
        if not did_demo_pretrain:
            wait_for_start()

    if args.test:
        env.set_eval_mode()
        agent.evaluate()
        logger.info("Evaluation finished")
    else:
        agent.run()
        logger.info("Training finished")


if __name__ == "__main__":
    main()
