import torch
import torch.nn as nn
import torch.nn.init as init


# initialize model weights
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        if m.bias:
            m.bias.data.zero_()


def compute_loss():
    pass

def create_vis_plot(vis, X_, Y_, title_, legend_):
    return vis.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=X_,
            ylabel=Y_,
            title=title_,
            legend=legend_
        )
    )