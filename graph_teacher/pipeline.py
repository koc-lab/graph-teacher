from tqdm.auto import tqdm

import wandb
from commons.early_stopper import EarlyStopper
from graph_teacher.trainer import Trainer


class Pipeline:
    def __init__(
        self,
        trainer: Trainer,
        max_epochs: int,
        patience: int,
        wandb_log_flag: bool = False,
    ):
        self.trainer = trainer
        self.max_epochs = max_epochs
        self.patience = patience
        self.wandb_log_flag = wandb_log_flag

    def run_single_configuration(self):
        t = tqdm(range(self.max_epochs), desc="Training Loop", leave=True)
        early_stopper = EarlyStopper(patience=self.patience, verbose=False)
        best_model = None
        for epoch in t:
            self.trainer.update_cls()
            e_loss = self.trainer.train_epoch()
            train_result = self.trainer.evaluate("train")
            val_result = self.trainer.evaluate("validation")
            if self.wandb_log_flag:
                print("Logging to wandb")
                self.wandb_log(e_loss, train_result, val_result)

            best_val_res, best_model = early_stopper(
                val_result, self.trainer.model, epoch
            )
            self.set_description(t, train_result, val_result, best_val_res)

            if early_stopper.early_stop:
                break
        self.best_model = best_model
        self.best_val_result = best_val_res

    @staticmethod
    def wandb_log(e_loss, train_result, val_result):
        wandb.log(
            {
                "e_loss": e_loss,
                "train_metric": train_result,
                "val_metric": val_result,
            }
        )

    def set_description(self, t: tqdm, train_result, val_result, best_val_res):
        t.set_description(
            f"Train {self.trainer.args.metric_name.value}: {train_result:.3f} | Validation {self.trainer.args.metric_name.value}: {val_result:.3f} | Best Validation {self.trainer.args.metric_name.value}: {best_val_res:.3f}"
        )
