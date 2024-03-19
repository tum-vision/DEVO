
import torch
from torch.utils.tensorboard import SummaryWriter


SUM_FREQ = 100 # TODO invariant to gpu_nums

class Logger:
    def __init__(self, name, scheduler, total_steps=0, step=1):
        self.total_steps = total_steps
        self.step = step
        self.running_loss = {}
        self.writer = None
        self.name = name
        self.scheduler = scheduler

    def _print_training_status(self):
        if self.writer is None:
            self.writer = SummaryWriter("../runs/{}".format(self.name))
            print([k for k in self.running_loss])

        lr = self.scheduler.get_lr().pop() # TODO use get_last_lr()
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in self.running_loss.keys()]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps * self.step + 1, lr)
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        for key in self.running_loss:
            val = self.running_loss[key] / SUM_FREQ
            # TODO all losses in one diagram (add_scalars)
            self.writer.add_scalar(key, val, self.total_steps * self.step)
            self.running_loss[key] = 0.0
        self.writer.add_scalar("lr", lr, self.total_steps * self.step)

    def push(self, metrics):

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter("../runs/{}".format(self.name))
            print([k for k in self.running_loss])
            
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps * self.step)

    def write_figures(self, figures):
        if self.writer is None:
            self.writer = SummaryWriter("../runs/{}".format(self.name))
            
        for key in figures:
            self.writer.add_figure(key, figures[key], self.total_steps * self.step)

    def close(self):
        self.writer.close()

