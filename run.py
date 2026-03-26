import os, sys, argparse, logging, time, signal
from datetime import datetime
from omegaconf import OmegaConf
import torch
import keyboard

os.environ['WANDB_MODE'] = 'offline'

from acEnv import ACEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Model"))
from Model.discor.algorithm import SAC, DisCor
from Model.discor.agent import Agent

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Common"))
from Common.logger import Logger

# Global agent for signal handler
global_agent = None
global_logger = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global global_agent, global_logger
    if global_logger:
        global_logger.info("\n=== TRAINING INTERRUPTED ===")
    if global_agent:
        try:
            log_dir = global_agent._log_dir if hasattr(global_agent, '_log_dir') else 'Outputs'
            checkpoint_path = os.path.join(log_dir, "checkpoint_interrupted")
            global_agent.save(checkpoint_path, save_buffer=False)
            if global_logger:
                global_logger.info(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            if global_logger:
                global_logger.error(f"Error saving checkpoint: {e}")
    if global_logger:
        global_logger.info("Resume with: python run.py --load_path <checkpoint_path>")
    sys.exit(0)

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
    global global_agent, global_logger
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    #region config, parseArgs into config
    parser = argparse.ArgumentParser(description="Autonomous Racing Agent Training")
    parser.add_argument("--config", default="config.yml", type=str, help="Path to configuration file")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load the model from")
    parser.add_argument("--algo", type=str, default="sac", help="Algorithm type (sac or discor)")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="key=value config overrides")
    
    args = parser.parse_args()
    if args.load_path is not None:
        args.load_path = os.path.abspath(args.load_path)
    
    # Load config.yml arguments first
    config = OmegaConf.load(args.config)

    # Add/Overwrite with command line arguments
    cli_conf = OmegaConf.from_dotlist(args.overrides)
    config = OmegaConf.merge(config, cli_conf)

    # Make a work directory
    if config.work_dir is not None:
        work_dir = os.path.abspath(config.work_dir)
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = "Outputs" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
        work_dir = os.path.abspath(work_dir) + os.sep
        os.makedirs(work_dir, exist_ok=True)
    config.work_dir = work_dir
    #endregion
    
    #region logger, SetupLogger
    loggingFormat = logging.Formatter('%(levelname)s:%(asctime)s:%(message)s')

    fileHandler = logging.FileHandler(config.work_dir + "log.log")
    fileHandler.setFormatter(loggingFormat)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(loggingFormat)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(stream_handler)
    
    global_logger = logger
    #endregion
    
    logger.info(f"\n{OmegaConf.to_yaml(config)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    env = ACEnv(config, logger)
    logger.info(f"Environment Initalized")
    
    # Wait for user to press spacebar before starting
    wait_for_start()

    if args.algo == 'discor':
        algo = DisCor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=config.seed,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            **OmegaConf.to_container(config.SAC), **OmegaConf.to_container(config.DisCor))
        logger.info(f"Using Algorithmn DisCor")
    elif args.algo == 'sac':
        algo = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=config.seed,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
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
    
    # Load checkpoint if provided
    if args.load_path:
        try:
            logger.info(f"Loading models from {args.load_path}")
            agent._algo.load_models(args.load_path)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.warning("Starting from scratch instead")
    
    global_agent = agent

    # Start training
    try:
        agent.run()
        logger.info("Done Training")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        signal_handler(None, None)
    finally:
        env.shutdown()
    
if __name__ == "__main__":
    main()
