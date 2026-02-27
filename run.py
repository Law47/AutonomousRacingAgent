import os, sys, argparse, logging, copy, time
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import torch
import keyboard

from acEnv import ACEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Model"))
from Model.discor.algorithm import SAC, DisCor
from Model.discor.agent import Agent

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Common"))
import Common.logging_config as logging_config
from Common.logger import Logger

def wait_for_start():
    """Wait for user to press spacebar before starting training"""
    print("\n" + "="*50)
    print("Environment ready!")
    print("Press SPACEBAR to start training...")
    print("="*50 + "\n")
    while True:
        try:
            if keyboard.is_pressed('space'):
                print("Starting training...\n")
                time.sleep(0.2)  # Debounce
                return
            time.sleep(0.05)
        except Exception as e:
            # If keyboard fails, start immediately
            print(f"Keyboard polling unavailable ({e}), starting immediately")
            return

def main():
    #region parseArgs into config
    parser = argparse.ArgumentParser(description="Description of your program.")
    parser.add_argument("--inference", default=False, type=bool, help="Run inference instead of training")
    parser.add_argument("--config", default="config.yml", type=str, help="Path to configuration file")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load the model from (default: None)")
    parser.add_argument("--algo", type=str, default="sac", help="Algorithm type (default: sac)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Any key=value arguments to override config values")
    
    args = parser.parse_args()
    args.load_path = os.path.abspath(args.load_path) + os.sep if args.load_path is not None else None
    
    # Load config.yml arguments first
    config = OmegaConf.load(args.config)

    # Add/Overwrite with command line arguments
    cli_conf = OmegaConf.from_dotlist(args.overrides)
    config = OmegaConf.merge(config, cli_conf)

    # Make a work directory if not specififed
    if config.work_dir is not None:
        work_dir = os.path.abspath(args.work_dir) + os.sep + config.track + os.sep + config.car + os.sep
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = "Outputs" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
        work_dir = os.path.abspath(work_dir) + os.sep
        os.makedirs(work_dir, exist_ok=True)
    config.work_dir = work_dir
    #endregion
    
    #region SetupLogger
    loggingFormat = logging.Formatter('%(levelname)s:%(asctime)s:%(message)s')

    fileHandler = logging.FileHandler(config.work_dir + "log.log")
    fileHandler.setFormatter(loggingFormat)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(loggingFormat)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(stream_handler)
    
    logger.info(f"Config: \n{OmegaConf.to_yaml(config)}")
    logger.info(f"Work Dir: {work_dir}")
    #endregion
    
    device = torch.device("cuda")
    assert device.type == "cuda", "Only cuda is supported"

    env = ACEnv(config, logger)
    logger.info(f"Environment Initalized")
    
    # Wait for user to press spacebar before starting
    wait_for_start()

    if args.algo == 'discor':
        algo = DisCor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=config.seed,
            **OmegaConf.to_container(config.SAC), **OmegaConf.to_container(config.DisCor))
        logger.info(f"Using Algorithmn DisCor")
    elif args.algo == 'sac':
        algo = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=config.seed,
            **OmegaConf.to_container(config.SAC))
        logger.info(f"Using Algorithmn SAC")
    else:
        raise Exception('You need to set algo sac or discor')

    # Update the logger configuration with dynamic values
    config.exp_name = f'{config.AssettoCorsa.car}-{config.AssettoCorsa.track}'
    config.action_dim = env.action_dim
    config.steps = config.Agent.num_steps

    # Initialize wandb logger
    if not config.disable_wandb:
        wandb_logger = Logger(config.copy())
    else:
        wandb_logger = None
        
    agent = Agent(env=env, test_env=env, algo=algo, log_dir=config.work_dir,
                  device=device, seed=config.seed, **config.Agent, wandb_logger=wandb_logger, logger=logger)
    

    # Start training
    agent.run()
    logger.info("Done Training")
    
if __name__ == "__main__":
    main()