import os
import argparse

from dotenv import load_dotenv
import wandb
from mmengine.config import Config
from mmengine.runner import Runner

import evaluator
import models
import process_data
import xray


def main():
    parser = argparse.ArgumentParser(description="Train a segmentor")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = "work-dirs"

    cfg.resume = False

    load_dotenv()
    wandb.login(key=os.getenv("mpark_API_KEY"))
    wandb.init(
        project=args.config.split("/")[-1],
        name=f"bs{cfg.train_dataloader.batch_size}_iter{cfg.train_cfg.max_iters}",
        entity="tjwlssla1-gachon-university",
    )

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
