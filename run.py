import os, sys, argparse, logging, copy
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(filename='log.log', encoding='utf-8', level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Description of your program.")
    parser.add_argument("--inference", default=False, type=bool, help="Run inference instead of training")
    parser.add_argument("--config", default="config.yml", type=str, help="Path to configuration file")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load the model from (default: None)")
    parser.add_argument("--algo", type=str, default="sac", help="Algorithm type (default: sac)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Any key=value arguments to override config values")
    
    args = parser.parse_args()
    args.load_path = os.path.abspath(args.load_path) + os.sep if args.load_path is not None else None
    return args

def main():
    args = parse_args()
    
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
        work_dir = "outputs" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
        work_dir = os.path.abspath(work_dir) + os.sep
        os.makedirs(work_dir, exist_ok=True)
    config.work_dir = work_dir


    logger.info(f"Config: {OmegaConf.to_yaml(config)}")
    logger.info(f"Work Dir: {work_dir}")
    
if __name__ == "__main__":
    main()