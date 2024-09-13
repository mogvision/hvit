import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


import argparse
import wandb
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

# Add this line after the imports
torch.set_float32_matmul_precision('medium')

from src import logger
from src.trainer import LiTHViT
from src.utils import read_yaml_file
from src.data.datasets import get_dataloader


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run training or inference")
    parser.add_argument("--num_gpus", type=int, default='1', help="Number of GPUs to use. Use '-1' for all available GPUs.")
    parser.add_argument("--experiment_name", type=str, default="OASIS", help="Experiment name")
    parser.add_argument("--mode", choices=["train", "inference"], default="train", help="Mode to run: train or inference")
    parser.add_argument("--train_data_path", type=str, default="/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/Mori/DATA/OASIS/OASIS_L2R_2021_task03/train", help="Path to the train set")
    parser.add_argument("--val_data_path", type=str, default="/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/Mori/DATA/OASIS/OASIS_L2R_2021_task03/test", help="Path to the validation set")
    parser.add_argument("--test_data_path", type=str, default="/home/mori/HViT/OASIS_small/test", help="Path to the test set")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model/checkpoint_path to load")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the best model")
    parser.add_argument("--mse_weights",type=float, default=1, help="MSE Loss weights")
    parser.add_argument("--dice_weights", type=float, default=1, help="Dice Loss weights")
    parser.add_argument("--grad_weights", type=float, default=0.02, help="Grad Loss weights")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--tgt2src_reg", type=bool, default=True, help="target to source registration during training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--max_epochs", type=int, default=1000, help="Maximum number of epochs")
    parser.add_argument("--num_labels", type=int, default=36, help="Number of labels")
    parser.add_argument("--precision", type=str, default='bf16', help="Precision")
    parser.add_argument("--hvit_light", type=bool, default=True, help="Use HViT-Light")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    config = read_yaml_file("./config/config.yaml")

    # Initialize a single WandbLogger instance
    wandb_logger = WandbLogger(project="wandb_HViT", name=args.experiment_name)

    # get dataloaders
    train_dataloader = get_dataloader(data_path = args.train_data_path, 
                                      input_dim=config["data_size"], 
                                      batch_size=args.batch_size,
                                      is_pair=False)

    val_dataloader = get_dataloader(data_path = args.val_data_path, 
                                    input_dim=config["data_size"], 
                                    batch_size=args.batch_size, 
                                    shuffle = False,
                                    is_pair=True)

    # Determine number of GPUs to use
    devices = min(int(args.num_gpus), torch.cuda.device_count()) if args.num_gpus > 0 else -1       
    print(f"Using {devices} GPUs ...")

    # setup trainer for specified number of GPUs
    trainer = Trainer(max_epochs=args.max_epochs, 
                      logger=[wandb_logger], 
                      precision=args.precision,
                      accelerator="gpu",
                      devices=devices,
                      strategy="ddp" if devices > 1 else "auto")  # Use "auto" for single GPU

    # train/test
    if args.mode == "train":
        if args.resume_from_checkpoint:
            model = LiTHViT.load_from_checkpoint(args.resume_from_checkpoint, args=args, wandb_logger=wandb_logger)
            print(f"Resuming training from epoch {model.last_epoch + 1}")
        else:
            model = LiTHViT(args, config, wandb_logger=wandb_logger)
            print("Starting new training run")
        logger.info("Starting training")
        trainer.fit(model, 
                    train_dataloaders=train_dataloader, 
                    val_dataloaders=val_dataloader, 
                    datamodule=None, 
                    ckpt_path=args.resume_from_checkpoint)

    elif args.mode == "inference":
        logger.info("Starting inference")

        test_dataloader = get_dataloader(data_path = args.test_data_path, 
                                input_dim=config["data_size"], 
                                is_pair=True, 
                                batch_size=args.batch_size, 
                                shuffle = False)


        # # Get the latest checkpoint folder
        # checkpoints_dir = Path("checkpoints")
        # checkpoints = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()], key=os.path.getctime, reverse=True)
        # latest_checkpoint = checkpoints[1] if checkpoints else None

        # if os.path.exists(latest_checkpoint):
        #     logger.info(f"Using latest checkpoint: {latest_checkpoint}")

        #     if args.checkpoint_path:
        #         best_model_path = f"{args.checkpoint_path}/best_model.ckpt"
        #     else:
        #         best_model_path = f"{latest_checkpoint}/best_model.ckpt"

        if args.checkpoint_path:
            model = LiTHViT.load_from_checkpoint(args.checkpoint_path, args=args, wandb_logger=wandb_logger)
            print(f"Checkpoint loaded. Resuming from epoch {model.last_epoch + 1}")
        else:
            raise Exception("No checkpoint found")
        trainer.test(model, dataloaders=test_dataloader)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
