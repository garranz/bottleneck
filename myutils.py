import torch as tc

def MIsandwich( mus: tc.Tensor, logvars: tc.Tensor ):
    '''
    Paramters
        mus         tc.Tensor (Nbatch, in_dim)
            posterior mean -> output from the enconder of a given input

        logvars     tc.Tensor (Nbatch, in_dim)
            posterior logvar -> output from the enconder of a given input

    Returns
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
