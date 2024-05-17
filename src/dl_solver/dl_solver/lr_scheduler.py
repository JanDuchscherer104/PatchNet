from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau, _LRScheduler
from torch.optim.optimizer import Optimizer


class HybridScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        monitor="train_loss",
        patience: int = 20,
    ):
        self.optimizer = optimizer
        self.cyclic_scheduler = CyclicLR(
            optimizer,
            base_lr=[group["lr"] for group in optimizer.param_groups],
            max_lr=[group["lr"] * 2 for group in optimizer.param_groups],
            step_size_up=step_size,
            mode="triangular2",
        )

        self.reduce_on_plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.8,  # Reduce LR by a factor of 0.5
            patience=1,
            threshold=0.01,  # Threshold for measuring new optimum
            cooldown=2,  # Number of epochs to wait before resuming normal operation after LR has been reduced
            min_lr=[group["lr"] * 1e-3 for group in optimizer.param_groups],
        )

        self.monitor = monitor
        self.patience = patience
        self.bad_steps = 0
        self.good_steps = 0
        self.current_scheduler = self.cyclic_scheduler
        self.prev_loss = float("inf")

        super().__init__(optimizer)

    def step(self, metrics=None, epoch=None):
        current_loss = metrics if metrics is not None else self.prev_loss
        if self.current_scheduler is self.cyclic_scheduler:
            if current_loss >= self.prev_loss:
                self.bad_steps += 1
            else:
                self.bad_steps = 0

            if self.bad_steps >= self.patience:
                self.bad_steps = 0
                self.current_scheduler = self.reduce_on_plateau_scheduler
                self.current_scheduler.last_epoch = self.cyclic_scheduler.last_epoch
        else:
            if current_loss < self.prev_loss:
                self.good_steps += 1
            else:
                self.good_steps = 0

            if self.good_steps >= self.patience:
                self.current_scheduler = self.cyclic_scheduler
                self.current_scheduler.last_epoch = (
                    self.reduce_on_plateau_scheduler.last_epoch
                )
                self.step()

        if self.current_scheduler is self.cyclic_scheduler:
            self.current_scheduler.step(epoch=epoch)
        else:
            self.current_scheduler.step(metrics=current_loss, epoch=epoch)

        self.prev_loss = current_loss

    def _cyclic_step(self):
        self.cyclic_scheduler.step()

    def get_last_lr(self):
        return self.current_scheduler.get_last_lr()

    def _get_closed_form_lr(self):
        return self.current_scheduler._get_closed_form_lr()
