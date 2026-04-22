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


def wait_for_start(expect_assetto_auto: bool = True) -> None:
    print("\n" + "=" * 50)
    print("Environment ready!")
    if expect_assetto_auto:
        print("Launch Assetto Corsa, enable its built-in auto shifter for the curriculum auto phase, then press SPACE to start training...")
    else:
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


def resolve_demonstration_config(config):
    top_level_demo = getattr(config, "Demonstrations", None)
    nested_demo = None
    assetto_cfg = getattr(config, "AssettoCorsa", None)
    if assetto_cfg is not None:
        nested_demo = getattr(assetto_cfg, "Demonstrations", None)

    if nested_demo is not None:
        if top_level_demo is not None:
            logger.warning(
                "Both Demonstrations and AssettoCorsa.Demonstrations are set; "
                "using AssettoCorsa.Demonstrations."
            )
        return nested_demo

    return top_level_demo


def maybe_load_demonstrations(agent: Agent, env, config) -> dict:
    demo_config = resolve_demonstration_config(config)
    if demo_config is None or not getattr(demo_config, "enabled", False):
        return {"loaded": False, "pretrained": False, "transitions": 0}

    data_paths = []
    single_path = getattr(demo_config, "data_path", None)
    if single_path:
        data_paths.append(single_path)
    for path in getattr(demo_config, "data_paths", []) or []:
        data_paths.append(path)

    if not data_paths:
        raise ValueError(
            "Demonstrations.enabled is true, but no data_path/data_paths were provided "
            "in Demonstrations or AssettoCorsa.Demonstrations"
        )

    logger.info("Demonstration paths requested: %s", data_paths)

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
        return {"loaded": True, "pretrained": True, "transitions": total_transitions}

    legacy_pretrain_steps = int(getattr(demo_config, "pretrain_steps", 0))
    if legacy_pretrain_steps > 0:
        logger.warning(
            "Demonstrations.pretrain_steps is deprecated. "
            "Use Demonstrations.pretrain_epochs so each epoch covers the demonstration dataset once."
        )
        agent.pre_train(legacy_pretrain_steps)
        return {"loaded": True, "pretrained": True, "transitions": total_transitions}

    logger.info("Demonstrations were loaded, but pretrain_epochs=0 so demo pretraining was skipped.")
    return {"loaded": True, "pretrained": False, "transitions": total_transitions}


def build_sac_kwargs(config):
    sac_kwargs = OmegaConf.to_container(config.SAC, resolve=True)
    shift_config = getattr(config, "ShiftModel", None)
    if shift_config is not None:
        shift_config = OmegaConf.to_container(shift_config, resolve=True)
        shift_key_map = {
            "enabled": "shift_enabled",
            "lr": "shift_lr",
            "entropy_lr": "shift_entropy_lr",
            "hidden_units": "shift_hidden_units",
            "loss_weight": "shift_loss_weight",
            "pos_weight": "shift_pos_weight",
            "threshold": "shift_threshold",
            "target_entropy": "shift_target_entropy",
            "reward_scale": "shift_reward_scale",
        }
        for source_key, target_key in shift_key_map.items():
            if source_key in shift_config:
                sac_kwargs[target_key] = shift_config[source_key]
    return sac_kwargs


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
        **build_sac_kwargs(config),
    )
    agent_kwargs = OmegaConf.to_container(config.Agent, resolve=True)

    agent = Agent(
        env=env,
        test_env=env,
        algo=algo,
        log_dir=config.work_dir,
        device=device,
        seed=config.seed,
        offline_sampling_config=OmegaConf.to_container(getattr(config, "OfflineSampling", {}), resolve=True),
        shift_curriculum_config=OmegaConf.to_container(getattr(config, "ShiftCurriculum", {}), resolve=True),
        wandb_logger=None,
        **agent_kwargs,
    )

    if args.load_path:
        load_path = os.path.abspath(args.load_path)
        if not load_path.endswith(os.sep):
            load_path += os.sep
        agent.load(load_path, load_buffer=(not args.test and not args.load_weights_only))

    if not args.test:
        if args.load_path:
            logger.info(
                "Skipping demonstration loading/pretraining because --load_path was provided; "
                "resuming from checkpoint instead."
            )
            demo_result = {"loaded": False, "pretrained": False, "transitions": 0}
        else:
            demo_result = maybe_load_demonstrations(agent, env, config)

        wait_for_start(expect_assetto_auto=agent.shift_ac_auto_phase_active())

    if args.test:
        env.set_eval_mode()
        agent.evaluate()
        logger.info("Evaluation finished")
    else:
        agent.run()
        logger.info("Training finished")


if __name__ == "__main__":
    main()
