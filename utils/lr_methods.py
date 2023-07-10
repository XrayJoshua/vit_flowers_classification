import torch


def warmup(optimizer, warm_up_iters, warm_up_factor):
    def f(x):
        if x >= warm_up_iters:
            return 1

        alpha = float(x) / warm_up_iters
        return warm_up_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
