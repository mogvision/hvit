import math
import torch
from lightning import LightningModule

from src import logger, checkpoints_dir
from src.model.hvit import HViT
from src.model.hvit_light import HViT_Light
from src.loss import loss_functions, DiceScore
from src.utils import get_one_hot

dtype_map = {
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
    'fp16': torch.float16
}

class LiTHViT(LightningModule):
    def __init__(self, args, config, wandb_logger=None, save_model_every_n_epochs=10):
        super().__init__()
        self.automatic_optimization = False
        self.args = args
        self.config = config
        self.best_val_loss = 1e8
        self.save_model_every_n_epochs = save_model_every_n_epochs
        self.lr = args.lr
        self.last_epoch = 0
        self.tgt2src_reg = args.tgt2src_reg
        self.hvit_light = args.hvit_light
        self.precision = args.precision

        self.hvit = HViT_Light(config) if self.hvit_light else HViT(config)

        self.loss_weights = {
            "mse": self.args.mse_weights,
            "dice": self.args.dice_weights,
            "grad": self.args.grad_weights
        }
        self.wandb_logger = wandb_logger
        self.test_step_outputs = []

    def _forward(self, batch, calc_score: bool = False, tgt2src_reg: bool = False):
        _loss = {}
        _score = 0.


        dtype_ = dtype_map.get(self.precision, torch.float32)

        with torch.amp.autocast(device_type="cuda", dtype=dtype_):
            if tgt2src_reg:
                target, source = batch[0].to(dtype=dtype_), batch[1].to(dtype=dtype_)
                tgt_seg, src_seg = batch[2], batch[3]
            else:
                source, target = batch[0].to(dtype=dtype_), batch[1].to(dtype=dtype_)
                src_seg, tgt_seg = batch[2], batch[3]
                
            moved, flow = self.hvit(source, target)

            if calc_score:
                moved_seg = self._get_one_hot_from_src(src_seg, flow, self.args.num_labels)
                _score = DiceScore(moved_seg, tgt_seg.long(), self.args.num_labels)

            _loss = {}
            for key, weight in self.loss_weights.items():
                if key == "mse":
                    _loss[key] = weight * loss_functions[key](moved, target)
                elif key == "dice":
                    moved_seg = self._get_one_hot_from_src(src_seg, flow, self.args.num_labels)
                    _loss[key] = weight * loss_functions[key](moved_seg, tgt_seg.long())
                elif key == "grad":
                    _loss[key] = weight * loss_functions[key](flow)
            
            _loss["avg_loss"] = sum(_loss.values()) / len(_loss)
        return _loss, _score

    def training_step(self, batch, batch_idx):
        self.hvit.train()
        opt = self.optimizers()
        
        loss1, _ = self._forward(batch, calc_score=False)
        self.manual_backward(loss1["avg_loss"])
        opt.step()
        opt.zero_grad()
            
        if self.tgt2src_reg:
            loss2, _ = self._forward(batch, tgt2src_reg=True, calc_score=False)
            self.manual_backward(loss2["avg_loss"])
            opt.step()
            opt.zero_grad()
        
        total_loss = {
            key: (loss1[key].item() + loss2[key].item()) / 2 if self.tgt2src_reg and key in loss2 else loss1[key].item()
            for key in loss1.keys()
        }

        self.wandb_logger.log_metrics(total_loss, step=self.global_step)
        return total_loss

    def on_train_epoch_end(self):
        if self.current_epoch % self.save_model_every_n_epochs == 0:
            checkpoint_path = f"{checkpoints_dir}/model_epoch_{self.current_epoch}.ckpt"
            self.trainer.save_checkpoint(checkpoint_path)
            logger.info(f"Saved model at epoch {self.current_epoch}")
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.wandb_logger.log_metrics({"learning_rate": current_lr}, step=self.global_step)


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.hvit.eval()
            _loss, _score = self._forward(batch, calc_score=True)
    
        # Log each component of the validation loss
        for loss_name, loss_value in _loss.items():
            self.log(f"val_{loss_name}", loss_value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    
        # Log the mean validation score if available
        if _score is not None:
            self.log("val_score", _score.mean(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    
        # Log to wandb
        log_dict = {f"val_{k}": v.item() for k, v in _loss.items()}
        log_dict.update({
            "val_score_mean": _score.mean().item() if _score is not None else None,
        })
        self.wandb_logger.log_metrics({k: v for k, v in log_dict.items() if v is not None}, step=self.global_step)
    
        return {"val_loss": _loss["avg_loss"], "val_score": _score.mean().item()}

    def on_validation_epoch_end(self):
        """
        Callback method called at the end of the validation epoch.
        Saves the best model based on validation loss and logs metrics.
        """
        val_loss = self.trainer.callback_metrics.get("val_loss")
        
        if val_loss is not None and self.current_epoch > 0:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = f"{checkpoints_dir}/best_model.ckpt"
                self.trainer.save_checkpoint(best_model_path)
                self.wandb_logger.experiment.log({
                    "best_model_saved": best_model_path,
                    "best_val_loss": self.best_val_loss.item()
                })
                logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step on a batch of data.
        
        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.
        
        Returns:
            A dictionary containing the test Dice score.
        """
        with torch.no_grad():
            self.hvit.eval()
            _, _score = self._forward(batch, calc_score=True)
    
        # Ensure _score is a tensor and take the mean
        _score = _score.mean() if isinstance(_score, torch.Tensor) else torch.tensor(_score).mean()
    
        self.test_step_outputs.append(_score)   

        # Log to wandb only if the logger is available
        if self.wandb_logger:
            self.wandb_logger.log_metrics({"test_dice": _score.item()}, step=self.global_step)

        # Return as a dict with tensor values
        return {"test_dice": _score}
    
    def on_test_epoch_end(self):
        """
        Callback method called at the end of the test epoch.
        Computes and logs the average test Dice score.
        """
        # Calculate the average Dice score across all test steps
        avg_test_dice = torch.stack(self.test_step_outputs).mean()

        # Log the average test Dice score
        self.log("avg_test_dice", avg_test_dice, prog_bar=True)

        # Log to wandb if available
        if self.wandb_logger:
            self.wandb_logger.log_metrics({"total_test_dice_avg": avg_test_dice.item()})

        # Clear the test step outputs list for the next test epoch
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for the model.
        
        Returns:
            A dictionary containing the optimizer and learning rate scheduler configuration.
        """
        optimizer = torch.optim.Adam(self.hvit.parameters(), lr=self.lr, weight_decay=0, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def lr_lambda(self, epoch):
        """
        Defines the learning rate schedule.
        
        Args:
            epoch: The current epoch number.
        
        Returns:
            The learning rate multiplier for the given epoch.
        """
        max_epochs = self.trainer.max_epochs
        return math.pow(1 - epoch / max_epochs, 0.9)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, args=None, wandb_logger=None):
        """
        Loads a model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
            args: Optional arguments to override saved ones.
            wandb_logger: Optional WandB logger instance.
        
        Returns:
            An instance of the model loaded from the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        args = args or checkpoint.get('hyper_parameters', {}).get('args')
        config = checkpoint.get('hyper_parameters', {}).get('config')
        
        model = cls(args, config, wandb_logger)
        model.load_state_dict(checkpoint['state_dict'])

        if 'hyper_parameters' in checkpoint:
            hyper_params = checkpoint['hyper_parameters']
            for attr in ['lr', 'best_val_loss', 'last_epoch']:
                setattr(model, attr, hyper_params.get(attr, getattr(model, attr)))

        return model

    def on_save_checkpoint(self, checkpoint):
        """
        Callback to save additional information in the checkpoint.
        
        Args:
            checkpoint: The checkpoint dictionary to be saved.
        """
        checkpoint['hyper_parameters'] = {
            'config': self.config,
            'lr': self.lr,
            'best_val_loss': self.best_val_loss,
            'last_epoch': self.current_epoch
        }

    def _get_one_hot_from_src(self, src_seg, flow, num_labels):
        """
        Converts source segmentation to one-hot encoding and applies deformation.
        
        Args:
            src_seg: Source segmentation.
            flow: Deformation flow.
            num_labels: Number of segmentation labels.
        
        Returns:
            Deformed one-hot encoded segmentation.
        """
        src_seg_onehot = get_one_hot(src_seg, self.args.num_labels)
        deformed_segs = [
            self.hvit.spatial_trans(src_seg_onehot[:, i:i+1, ...].float(), flow.float())
            for i in range(num_labels)
        ]
        return torch.cat(deformed_segs, dim=1)

