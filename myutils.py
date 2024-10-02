import torch as tc
import torch.nn as nn
from typing import Type
import numpy as np

def mlp( din: int, dout: int, arch: tuple, 
        fn_act: Type[nn.Module] )->nn.Sequential:
    '''Creates an feed-forward network (or multi-layer perceptron) with the
    specified number of inputs, outputs, neurons per layer and activation
    function.

    Parameters
        din         int
            Number of inputs for the first layer

        dout        int
            Number of outputs

        arch        tuple
            (neurons_l1, neurons_l2, ... neurons_lL) number of neurons per
            layer. len(arch) == Number of layers

        fn_act      nn.Module
            Activation function for the neurons.

    Returns
        nn.Sequential
            ANN with the specify architecture
    '''
    # Input Layer
    layers = []

    ## Positional Encoding
    #if use_positional_encoding:
    #    layers.append(PositionalEncoding(positional_encoding_frequencies))

    # Hidden Layers
    N0 = din
    for N1 in arch:
        layers += [ nn.Linear( N0, N1 ), fn_act() ]
        N0 = N1
    layers += [ nn.Linear( N0, dout ) ]

    return nn.Sequential(*layers) 


def MIsandwich( mus: tc.Tensor, logvars: tc.Tensor )-> tuple[tc.Tensor,
    tc.Tensor]:
    '''Compute the bounds for the Mutual information when the mus and logvars
    for the posterior distribution are known. This method is intended to be used
    by DIBnet. In particular, the output of the enconders of DIBnet are the
    input to MIsandwich.

    Paramters
        mus         tc.Tensor (Nbatch, in_dim)
            posterior mean -> output from the enconder of a given input

        logvars     tc.Tensor (Nbatch, in_dim)
            posterior logvar -> output from the enconder of a given input

    Returns
        tc.Tensor (1,)
            INCE lower bound

        tc.Tensor (1,)
            Leave-1-Out upper bound
    '''
    # Convert varibales to double for extra precision
    mus_d, logvars_d = mus.double(), logvars.double()
    stds_d = tc.exp(logvars_d/2.)

    u_samples = tc.normal(mean=mus_d, std=stds_d )

    # Compute pairwise distance between sample points ant the centers of the
    # conditional distributions
    # (Nsamples,Nsamples,in_dim) = (Nsamples,1,in_dim) - (1,Nsamples,in_dim)
    d_ui_muj = u_samples.unsqueeze(1) - mus_d.unsqueeze(0)
    d_ui_muj /= stds_d.unsqueeze(0)  # Normalize by std

    p_ui_xj = tc.exp(tc.sum(d_ui_muj**2,dim=-1)/2. - 
        tc.sum(logvars_d,dim=-1).unsqueeze(0)/2.)
    # NOTE: for a multivariate gaussian distribution:
    # p(ui| muj, sj ) = 1/ [ (2pi)^(k/2) * |S|^1/2 ] * exp( 1/2 * d_ui_muj^2 )
    # the last term in p_ui_xj comes from inserting |S|^1/2 into the exponent
    p_ui_xj /= (2.*tc.pi)**( mus_d.size(-1) / 2. )
    # the sum of the square acts as the dot product ( u - mu )T ( u - mu )

    ############################################################################
    # InfoNCE (lower bound)
    ############################################################################
    # Extract diagonal elements (p_ui_xi)
    p_ui_xi = tc.diagonal(p_ui_xj, dim1=-2, dim2=-1)

    # Compute the InfoNCE lower bound
    infonce_lower = tc.mean( tc.log(p_ui_xi / tc.mean(p_ui_xj,dim=1)) )


    ############################################################################
    # Leave 1 Out (upper bound)
    ############################################################################
    # Zero out diagonal elements of p_ui_cond_xj
    I = tc.eye(mus_d.size(0), dtype=tc.double, device=mus.device)
    p_ui_xj_noD = p_ui_xj * (1. - I) 
    # we have to create a new matrix, if we use *=, p_ui_xi is also
    # updated (making it 0)

    # Compute the Leave-One-Out upper bound
    l1o_upper = tc.mean(tc.log(p_ui_xi / tc.mean(p_ui_xj_noD, dim=1)))

    return infonce_lower, l1o_upper


# These functions have been copied from testboundsMI. Note that we make slights
# modifications to accomodate for the new definition of mlp
class SeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dimX: int, dimY: int, dimE: int, 
                 arcX: tuple, arcY: tuple, fn_act: Type[nn.Module]):
        super(SeparableCritic, self).__init__()
        self._g = mlp(dimX, dimE, arcX, fn_act)
        self._h = mlp(dimY, dimE, arcY, fn_act)

    def forward(self, x: tc.Tensor, y: tc.Tensor):
        return tc.matmul(self._h(y), self._g(x).t())


class ConcatCritic(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the 
    value."""

    def __init__(self, dimX: int, dimY: int, arc: tuple, 
                 fn_act: Type[nn.Module]):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp( dimX+dimY, 1, arc, fn_act )

    def forward(self, x: tc.Tensor, y: tc.Tensor):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = tc.stack([x] * batch_size, dim=0)
        y_tiled = tc.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = tc.reshape(tc.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return tc.reshape(scores, [batch_size, batch_size]).t()


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, arch):
        super(MINE, self).__init__()
        self.T_func = mlp( x_dim + y_dim, 1, arch, nn.ReLU )

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = tc.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(tc.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(tc.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - tc.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, arch):
        super(InfoNCE, self).__init__()

        self.F_func = mlp( x_dim + y_dim, 1, arch, nn.ReLU )
        self.F_func.append( nn.Softplus() )

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(tc.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(tc.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)



def logmeanexp_nodiag(x, device='cuda'):
    batch_size = x.size(0)
    dim = (0, 1)

    logsumexp = tc.logsumexp(
        x - tc.diag(tc.inf * tc.ones(batch_size).to(device)), dim=dim)

    num_elem = batch_size * (batch_size - 1.)

    return logsumexp - tc.log(tc.tensor(num_elem)).to(device)


def mine_lower_bound(f, momentum=0.9,device="cuda"):
    """
    MINE lower bound based on DV inequality. 
    """
    first_term = f.diag().mean()
    buffer_update = logmeanexp_nodiag(f, device).exp()

    with tc.no_grad():
        second_term = logmeanexp_nodiag(f, device)
        buffer_new = momentum + buffer_update * (1 - momentum)
        buffer_new = tc.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new

    third_term_grad = buffer_update / buffer_new

    return first_term - second_term - third_term_grad + third_term_no_grad, buffer_update

def infonce_lower_bound(scores, device="cuda"):
    nll = scores.diag().mean() - scores.logsumexp(dim=1)
    # Alternative implementation:
    # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
    mi = tc.tensor(scores.size(0)).float().log() + nll
    mi = mi.mean()
    return mi

