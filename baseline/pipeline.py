from tqdm.auto import tqdm

from baseline.trainer import Trainer
from commons.early_stopper import EarlyStopper


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
        t = tqdm(range(self.max_epochs))
        early_stopper = EarlyStopper(patience=self.patience, verbose=False)
        best_model = None
        for epoch in t:
            e_loss = self.trainer.train_epoch()
            train_result = self.trainer.evaluate("train")
            val_result = self.trainer.evaluate("validation")
            if self.wandb_log_flag:
                self.wandb_log()
            self.set_description(e_loss, train_result, val_result)

            best_val_res, best_model = early_stopper(
                val_result, self.trainer.model, epoch
            )

            if early_stopper.early_stop:
                break
        self.best_model = best_model
        self.best_val_result = best_val_res

    @staticmethod
    def wandb_log():
        pass

    @staticmethod
    def set_description():
        pass
