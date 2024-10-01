import torch as tc
import torch.nn as nn
from .myutils import mlp


class DIBnet( nn.Module ):

    def __init__( self, in_dims: tuple, out_dim: int, Earch: tuple= (8,8), 
                 in_emb_d: int= 32, Oarch: tuple= (8,4), 
                 out_fn_act = nn.ReLU ):
        '''
        Parameters
            in_dims     list | tuple (ints)
                the i-entry correspond to the number of dimensions of the
                i-input. Thus, len( in_dims ) is the number of inputs.

            out_dim:    int
                number of output dimensions

            Earch       list | tuple (ints) (*(8,8))
                Architecture of the encoder. Assume all encoders are the same
                for all input. List with the number of neurons 
                for each layer. e.g.: an encoder with 12 neurons in the first
                layer and 24 in the second layer -> (12,24)

            in_emb_d    int (*32)
                embedding space dimensionality for each input (same for each
                feature)

            Oarch       list | tuple (ints) (*(8,4))
                Architecture of the mlp that tries to reconstruct the mutual
                information with the output. List with the number of neurons 
                for each layer. e.g.: an encoder with 12 neurons in the first
                layer and 24 in the second layer -> (12,24)

            out_fn_act  nn.modules.activation   (*nn.ReLU)
                Activation function for the final output layer
        '''
        super(DIBnet, self).__init__()  # Calling the parent class constructor

        self.in_dims = in_dims       # number of dimensions for each input
        self.Nin = len( in_dims )    # number of inputs
        self.in_emb_d = in_emb_d     # number of embedded dimension of input

        ########################################################################
        # Create encoders for each input
        ########################################################################
        fn_act = nn.ReLU    # Activation function

        Elist = []
        for in_d in self.in_dims:

            # NOTE: The last layer returns the means and stds of p(u|x), so the 
            # number of outputs is 2*in_emb_d
            Elist.append( mlp( in_d, 2*self.in_emb_d, Earch, fn_act ) )

        self.Elist = Elist # list of encoders (one for each input)

        ########################################################################
        # Create integration network
        ########################################################################
        self.OutNet = mlp( self.in_emb_d*self.Nin, out_dim, Oarch, fn_act )
        TODO: chaaaange
        #self.OutNet.append( out_fn_act() )


    def forward(self, inputs: tuple ):
        '''
        Parameters
            inputs      tuple
                Dimension: self.Nin. The i-item in the list is a tc.Tensor 
                of size (Nsmaples, self.in_dims[i])

        Returns
            tc.tensor   ( Nsamples, out_dim )
                Prediction from the outer net using the input embeddings

            tuple       ( Nin, )
                average KL for each input
        '''
        emb_ch_all = []
        kl_all = []

        # For each input, apply the enconder
        for k,inp in enumerate(inputs):
            # Split the output of the encoder into the mean and logvar of each
            # embeded feature -> (mu, log[std**2] )
            emb_mu, emb_logvar = tc.split( self.Elist[k](inp), 
                                          self.in_emb_d, dim=-1)

            # For each sample (mu,std), return a random number that follows that
            # normal distribution
            emb_ch = tc.normal(mean=emb_mu, std=tc.exp(emb_logvar/2.))
            # NOTE -> emb_ch.shape = ( Nsamples, in_emb_d )
            emb_ch_all.append( emb_ch )

            # Compute the KL wrt N(0,1) for every sample, then take the average:
            kl = tc.mean( 0.5*(tc.square(emb_mu) + 
                tc.exp(emb_logvar) - emb_logvar - 1.) )
            kl_all.append( kl )

        # The input for OutNet must be ( Nsamples, in_emb_d*Nin )
        return self.OutNet( tc.cat( emb_ch_all, -1 ) ), kl_all


