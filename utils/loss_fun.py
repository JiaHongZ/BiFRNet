import torch
import torch.nn as nn

class Var_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, mean, var, pre, lab):
        # ---> if [y] is not provided but [scores] is, calculate variational loss using weighted sum of prior-modes
        variatL = calculate_variat_loss(z=z, mu=mean, logvar=var, y=lab, y_prob=pre)
        variatL = lf.weighted_average(variatL, weights=batch_weights, dim=0)  # -> average over batch
        variatL /= (self.image_channels * self.image_size ** 2)  # -> divide by # of input-pixels
        return variatL

