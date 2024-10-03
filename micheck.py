import numpy as np
import torch 
import torch.nn as nn

from mymodels import DIBnet
from myutils import ConcatCritic, infonce_lower_bound, mine_lower_bound, MINE, InfoNCE

import matplotlib.pyplot as plt

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, cubic=None):
    """Generate samples from a correlated Gaussian distribution."""
    x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps

    if cubic is not None:
        y = y ** 3

    return x, y


def rho_to_mi(dim, rho):
    """Obtain the ground truth mutual information from rho."""
    return -0.5 * np.log(1 - rho**2) * dim


def mi_to_rho(dim, mi):
    """Obtain the rho for Gaussian give ground truth mutual information."""
    return np.sqrt(1 - np.exp(-2.0 / dim * mi))


if __name__ == "__main__":

    #crit = ConcatCritic(2,2,(32,16),nn.ReLU)
    #optimizer = torch.optim.Adam(crit.parameters(), 5e-4) # pyright:ignore

    #model = InfoNCE(2,2,(32,16))
    model = MINE(2,2,(32,16))
    optimizer = torch.optim.Adam(model.parameters(), 5e-4) # pyright:ignore

    mi_est_values = []

    #for i in range(1000):

    #    x, y = sample_correlated_gaussian(.5, dim=2, batch_size=512)


    #    crit.eval()
    #    mi_est_values.append(infonce_lower_bound( crit(x, y) ).item())

    #    crit.train() 

    #    model_loss = -infonce_lower_bound( crit( x, y ) )

    #    optimizer.zero_grad()
    #    model_loss.backward()
    #    optimizer.step()

    #    print( i )


    models = ( MINE(2,2,(32,16)), InfoNCE(2,2,(32,16)) )
    for model in models:
        mi_est_values = []
        optimizer = torch.optim.Adam(model.parameters(), 5e-4) # pyright:ignore
        for i in range(500):

            x, y = sample_correlated_gaussian(.5, dim=2, batch_size=1024)


            model.eval()
            mi_est_values.append( -model.learning_loss(x,y).item())

            model.train() 

            model_loss = model.learning_loss( x, y )

            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            print( i )

        plt.plot( mi_est_values )
