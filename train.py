import argparse
import logging
import os
import sys
import time
from datetime import datetime

import torch
from omegaconf import OmegaConf

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

sys.path.extend([
    os.path.join(ROOT_DIR, "assetto_corsa_gym"),
    os.path.join(ROOT_DIR, "algorithm", "discor"),
])

import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from discor.agent import Agent
from discor.algorithm.sac import SAC
import common.logging_config as logging_config
import common.misc as misc

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

    wait_for_start()

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
        agent.load(load_path, load_buffer=not args.test)

    if args.test:
        env.set_eval_mode()
        agent.evaluate()
        logger.info("Evaluation finished")
    else:
        agent.run()
        logger.info("Training finished")


if __name__ == "__main__":
    main()
