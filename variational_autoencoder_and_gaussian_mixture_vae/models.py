import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.distributions as dist
import pandas as pd
import torch.nn.functional as F


## VAE

class VaeInference(nn.Module):
    
    """
    VAE Inference block as described in the paper
    """
    
    def __init__(self, input_size, latent_dim, accumulated_kl_div, encoder_hidden_sizes=[]):
        """
        Initialize the variational parameters for weight and bias with nn.Parameter:
        Mean should be initialised to zeros and rho to random.

        Args:
        - input_size (int): The size of the input features.
        - latent_dim (int): The dimension of the latent variable.
        - accumulated_kl_div (numpy.ndarray): Accumulated KL divergence for tracking during training.
        - encoder_hidden_sizes (list): List of integers representing the number of units for hidden layers.
        """
        super().__init__()
        self.accumulated_kl_div = accumulated_kl_div
        
        # Initialize lists for mean and rho parameters
        w_mu_list, b_mu_list, w_rho_list, b_rho_list = [], [], [], []
        a1 = input_size

        # loop through the encoder hidden sizes to createt the parameters for each layer
        for layer_dim in encoder_hidden_sizes:
            w_mu_list.append(nn.Parameter(torch.zeros(a1, layer_dim)))
            b_mu_list.append(nn.Parameter(torch.zeros(layer_dim)))
            w_rho_list.append(nn.Parameter(torch.rand(a1, layer_dim)))
            b_rho_list.append(nn.Parameter(torch.rand(layer_dim)))
            a1 = layer_dim
        
        # parameters for the  the last latent layer
        w_mu_list.append(nn.Parameter(torch.zeros(a1, latent_dim)))
        b_mu_list.append(nn.Parameter(torch.zeros(latent_dim)))
            
        w_rho_list.append(nn.Parameter(torch.rand(a1, 1)))
        b_rho_list.append(nn.Parameter(torch.rand(1)))
            
        self.w_mu = nn.ParameterList(w_mu_list)
        self.b_mu = nn.ParameterList(b_mu_list)
        self.w_rho = nn.ParameterList(w_rho_list)
        self.b_rho = nn.ParameterList(b_rho_list)
        
        batch_norm_mu_list = []
        batch_norm_rho_list = []
        if len(encoder_hidden_sizes) > 0:
            for layer_dim in encoder_hidden_sizes:
                batch_norm_mu_list.append(nn.BatchNorm1d(num_features=layer_dim))
                batch_norm_rho_list.append(nn.BatchNorm1d(num_features=layer_dim))
        
        self.batch_norm_mu = nn.ModuleList(batch_norm_mu_list)
        self.batch_norm_rho = nn.ModuleList(batch_norm_rho_list)

        self.relu = nn.ReLU()

    def sampling(self, mu, rho, n_samples):
        """
        Sample weights using the reparameterization trick.

        Args:
        - mu (torch.Tensor): Mean parameter.
        - rho (torch.Tensor): Rho parameter.
        - n_samples (int): Number of samples.

        Returns:
        - torch.Tensor: Sampled latent variable using reparameterization trick.
        """
        sigma = torch.log(1 + torch.exp(rho))
        return mu.to('cuda' if torch.cuda.is_available() else 'cpu') + sigma*(torch.randn(n_samples, mu.shape[0])).to('cuda' if torch.cuda.is_available() else 'cpu')

    def kl_divergence(self, z, mu_theta, rho_theta, prior_sd=1):
        """
        Compute the KL-divergence term for weight parameters between the prior distribution of z and its posterior.

        Args:
        - z (torch.Tensor): Latent variable.
        - mu_theta (torch.Tensor): Mean parameter.
        - rho_theta (torch.Tensor): Rho parameter.
        - prior_sd (float): Standard deviation of the prior distribution.

        Returns:
        - torch.Tensor: KL divergence term.
        """

        log_prior = dist.Normal(0, prior_sd).log_prob(z)
        log_p_q = dist.Normal(mu_theta, torch.log(1 + torch.exp(rho_theta))).log_prob(z)
        
        return (log_p_q - log_prior).sum() / z.shape[0] 

    def forward(self, x):
        """
        Forward pass of the VAE Inference block.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Sampled latent variable.
        """
        
        mu = x.clone()
        for num_layer, w_mu, b_mu in zip(range(len(self.w_mu)-1), self.w_mu[:-1], self.b_mu[:-1]):
            mu = self.relu(mu.matmul(w_mu) + b_mu)
            # print('mu just before batch norm',mu.shape)
            if len(self.batch_norm_mu) > 0:
                mu = self.batch_norm_mu[num_layer](mu)
                # print('mu jsut after batch norm', mu.shape)
                
        mu = mu.matmul(self.w_mu[-1]) + self.b_mu[-1] # last hidden inference layer, without relu and batch norm
        # print('mu', pd.DataFrame(torch.max(mu, dim=1).values.reshape((mu.shape[0],1)).detach().cpu().numpy()).describe())
        rho = x.clone()
        for num_layer, w_rho, b_rho in zip(range(len(self.w_rho)), self.w_rho, self.b_rho):
            rho = self.relu(rho.matmul(w_rho) + b_rho)
            rho = torch.min(rho, torch.tensor(100))
            
            if len(self.batch_norm_rho) > 0:
                if num_layer < len(self.batch_norm_rho): # no batch norm in the last layer
                    rho = self.batch_norm_rho[num_layer](rho)
         
        # print('rho', pd.DataFrame(rho.detach().cpu().numpy()).describe())
        

        z = self.sampling(mu.mean(dim = 0), rho.mean(dim = 0), n_samples = x.shape[0])
        
        # Compute KL-div loss for training
        self.accumulated_kl_div[0] += self.kl_divergence(z, mu.mean(dim = 0), rho.mean(dim = 0))
        # print('in variational inference', z.shape)
        return z

class VaeGenerationLayer(nn.Module):
    """
    VAE generation block as described in the paper
    """
    
    def __init__(self, latent_dim, output_size, dist_name,mask_x0= None, eps = 1e-6):
        """
        Initialize VAE generation layer.

        Args:
        - latent_dim (int): Dimension of the latent variable.
        - output_size (int): Size of the output features.
        - dist_name (str): Name of the distribution ('Poisson', 'NB', 'ZIP', 'ZINB').
        - mask_x0 (torch.Tensor): Mask for handling zero values in the input.
        - eps (float): Small constant to avoid division by zero.
        """
        super().__init__()
        self.fc = nn.Linear(latent_dim, output_size, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid= nn.Sigmoid()
        self.dist_name= dist_name
        self.eps = eps
        
    
    def forward(self, z, mask_x0):
        """
        Forward pass of the VAE Generation Layer.

        Args:
        - z (torch.Tensor): Latent variable.
        - mask_x0 (torch.Tensor): Mask for handling zero values in the input.

        Returns:
        - torch.Tensor or tuple: Output tensor(s) based on the distribution.
        """
        if self.dist_name == 'Poisson':
            # For Poisson distribution, apply ReLU activation to ensure non-negative values
            lambda_ = self.relu(self.fc(z))
            # Add a small constant to ensure non-zero values (avoiding division by zero)
            return lambda_ + self.eps
        elif self.dist_name == 'NB':
            # For Negative Binomial (NB) distribution, apply sigmoid to ensure values between 0 and 1
            alpha = self.sigmoid(self.fc(z))
            beta = self.sigmoid(self.fc(z))
            return alpha, beta
        elif self.dist_name == 'ZIP':
            # For Zero-Inflated Poisson (ZIP) distribution, apply sigmoid to ensure values between 0 and 1
            psi = self.sigmoid(self.fc(z))
            lambda_ = self.sigmoid(self.fc(z))
            return psi, lambda_ 
        
        elif self.dist_name == 'ZINB':
            # For Zero-Inflated Negative Binomial (ZINB) distribution,
            # apply sigmoid to ensure values between 0 and 1 for pi, p, and ReLU for log_r
            pi= self.sigmoid(self.fc(z))
            p= self.sigmoid(self.fc(z))
            log_r= self.relu(self.fc(z))
            return pi, p, log_r
        else:
            print('distribution not recognized')
            return
    
class VariationalAutoEncoder(nn.Module):
    """
    VAE model
    """
    def __init__(self, input_size, latent_dim, encoder_hidden_sizes=[], decoder_hidden_sizes= [], dist_name= 'Poisson', return_latent= 'False'):
        """
        Initialize VAE model.

        Args:
        - input_size (int): Size of the input features.
        - latent_dim (int): Dimension of the latent variable.
        - encoder_hidden_sizes (list): List of integers representing the number of units for encoder hidden layers.
        - decoder_hidden_sizes (list): List of integers representing the number of units for decoder hidden layers.
        - dist_name (str): Name of the distribution ('Poisson', 'NB', 'ZIP', 'ZINB').
        - return_latent (str): Return latent variable during forward pass ('True' or 'False').
        """
        super().__init__()
        self.dist_name= dist_name
        self.accumulated_kl_div = np.zeros(1)
        
        self.return_latent= return_latent
        self.encoder = VaeInference(input_size, latent_dim, self.accumulated_kl_div, encoder_hidden_sizes)
        
        decoder_list = []
        if len(decoder_hidden_sizes)>0:
            a1 = latent_dim
            for hidden_size in decoder_hidden_sizes: # adding hidden layers
                decoder_list.append(VaeGenerationLayer(latent_dim=a1, output_size=hidden_size, dist_name= self.dist_name))
                a1 = hidden_size   
            decoder_list.append(VaeGenerationLayer(latent_dim=a1, output_size=input_size, dist_name= self.dist_name)) # output layer
        else:
            decoder_list.append(VaeGenerationLayer(latent_dim=latent_dim, output_size=input_size, dist_name= self.dist_name))
            
        self.decoder = nn.ModuleList(decoder_list)
        
        batch_norm_list = []
        if len(decoder_hidden_sizes) > 0:
            for layer_dim in decoder_hidden_sizes:
                batch_norm_list.append(nn.BatchNorm1d(num_features=layer_dim))
        
        self.batch_norm = nn.ModuleList(batch_norm_list)

    def reset_kl_div(self):
        """
        Reset the accumulated KL divergence.
        """
        self.accumulated_kl_div[0] = 0

    def forward(self, x):
        """
        Forward pass of the VAE model.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor or tuple: Output tensor(s) based on the distribution.
        """
        
        out = self.encoder(x)
        latent_var= out
        mask_x0 = x==0
        for num_layer, layer in enumerate(self.decoder):
          if not isinstance(out, tuple): #Poisson with one parameter
            out = layer(out, mask_x0)

          elif len(out)==2:
            # out= tuple([layer(param) for param in out])
            out1, out2 = out
            out1,_= layer(out1, mask_x0)
            _,out2= layer(out2, mask_x0)
            out = tuple([out1, out2])
          elif len(out)==3: #ZINB has 3params
            
            out1, out2, out3 = out
            out1,_,_= layer(out1, mask_x0)
            _,out2,_= layer(out2, mask_x0)
            _,_,out3=layer(out3, mask_x0)
            out = tuple([out1, out2,out3])
            if len(self.batch_norm) > 0:
                if num_layer < len(self.batch_norm): # no batch norm in the last layer
                    if isinstance(out, tuple):
                      out = tuple([self.batch_norm[num_layer](par) for par in out])
                  
                    else:
                      out = self.batch_norm[num_layer](out)
        desc = None
        if self.dist_name== 'Poisson':
          try:
            desc= pd.DataFrame(torch.tensor([[out[i][mask_x0[i]].max()] for i in range(out.shape[0])]).detach().cpu().numpy()).describe()
          except:
            True
        
        
        if not isinstance(out, tuple):
          lambda_ = (out).mean(dim=0)
        else:
          lambda_ =tuple([par.mean(dim=0) for par in out])
        if self.return_latent:
          return lambda_ , latent_var, desc
        else:
          return lambda_ , None, desc




#### GMVAE
class GMVaeInference(nn.Module):
    
    """
    Gaussian mixture variational autoencoder inference block as described in the paper

    Parameters:
        input_size (int): The size of the input data.
        latent_dim (int): The dimensionality of the latent space.
        nb_cat (int): The number of categories in the categorical distribution.
        accumulated_kl_div_y (list): A list to accumulate KL divergence values for categorical variables.
        accumulated_kl_div_z (list): A list to accumulate KL divergence values for latent variables.
        prior_pi (tensor, optional): Prior distribution for categorical variables. Defaults to None.
        encoder_hidden_sizes (list, optional): List of hidden layer dimensions for the encoder network. Defaults to [].
        pi_hidden_sizes (list, optional): List of hidden layer dimensions for the categorical distribution network. Defaults to [].
        train_mode (bool, optional): Flag indicating whether the model is in training mode. Defaults to True.
    """

    def __init__(self, 
                 input_size,
                 latent_dim, 
                 nb_cat,
                 accumulated_kl_div_y, 
                 accumulated_kl_div_z,
                 prior_pi = None,
                 encoder_hidden_sizes = [],
                 pi_hidden_sizes = [],
                 train_mode = True):
        
        super().__init__()
        self.accumulated_kl_div_z = accumulated_kl_div_z
        self.accumulated_kl_div_y = accumulated_kl_div_y
        self.nb_cat = nb_cat
        self.prior_pi = prior_pi
        self.train_mode = train_mode

        w_pi_list = []
        b_pi_list = []
        a1 = input_size
        for layer_dim in pi_hidden_sizes:
            w_pi_list.append(nn.Parameter(torch.rand(a1, layer_dim)))
            b_pi_list.append(nn.Parameter(torch.rand(layer_dim)))
            a1 = layer_dim
         
        w_pi_list.append(nn.Parameter(torch.rand(a1, self.nb_cat)))
        b_pi_list.append(nn.Parameter(torch.rand(self.nb_cat)))
        
        self.w_pi = nn.ParameterList(w_pi_list)
        self.b_pi = nn.ParameterList(b_pi_list)
        
        self.softmax = F.softmax
        
        w_mu_list = []
        b_mu_list = []
        w_rho_list = []
        b_rho_list = []
        a1 = input_size+ self.nb_cat
        for layer_dim in encoder_hidden_sizes:
            w_mu_list.append(nn.Parameter(torch.zeros(a1, layer_dim)))
            b_mu_list.append(nn.Parameter(torch.zeros(layer_dim)))
            w_rho_list.append(nn.Parameter(torch.rand(a1, layer_dim)))
            b_rho_list.append(nn.Parameter(torch.rand(layer_dim)))
            a1 = layer_dim
        
        w_mu_list.append(nn.Parameter(torch.zeros(a1, latent_dim)))
        b_mu_list.append(nn.Parameter(torch.zeros(latent_dim)))
            
        w_rho_list.append(nn.Parameter(torch.rand(a1, 1)))
        b_rho_list.append(nn.Parameter(torch.rand(1)))

        self.w_mu = nn.ParameterList(w_mu_list)
        self.b_mu = nn.ParameterList(b_mu_list)
        
        self.w_rho = nn.ParameterList(w_rho_list)
        self.b_rho = nn.ParameterList(b_rho_list)
        
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.2)

    def sampling_z(self, mu, rho, n_samples):
        
        """
        Sample weights using the reparametrization trick.
        
        Parameters:
            mu (tensor): Mean of the distribution.
            rho (tensor): Log of the standard deviation of the distribution.
            n_samples (int): Number of samples to generate.

        Returns:
            tensor: Sampled values using the reparametrization trick.
        """
        
        sigma = torch.log(1 + torch.exp(rho))
        
        return mu + sigma*(torch.randn(n_samples, mu.shape[0]))
    
    def sampling_y(self, pi, n_samples):

        """
        Sample categorical variables.
        
        Parameters:
            pi (tensor): Categorical distribution probabilities.
            n_samples (int): Number of samples to generate.

        Returns:
            tuple: Tuple containing the sampled categorical variable and its one-hot encoding.
        """

        y = dist.Categorical(pi).sample((n_samples,))
        one_hot_y = torch.eye(len(pi))[y]
        
        return y, one_hot_y


    def kl_divergence_y(self, y, pi, prior_pi):
        
        """
        Compute the KL-divergence term for categorical variables.
        
        Parameters:
            y (tensor): Categorical variable.
            pi (tensor): Posterior distribution probabilities.
            prior_pi (tensor): Prior distribution probabilities.

        Returns:
            tensor: KL-divergence value.
        """

        if prior_pi == None:
            prior_pi = torch.ones(len(pi), requires_grad=True)/self.nb_cat
    
        log_prior = dist.Categorical(prior_pi).log_prob(y)
        log_p_q = dist.Categorical(pi).log_prob(y) # use directly pi with the new appraoch
        
        return (log_p_q - log_prior).sum() / y.shape[0] 
    
    def kl_divergence_z(self, z, mu_theta, rho_theta, prior_sd=1):
        
        """
        Compute the KL-divergence term for latent variables.
        
        Parameters:
            z (tensor): Latent variable.
            mu_theta (tensor): Mean of the posterior distribution.
            rho_theta (tensor): Log of the standard deviation of the posterior distribution.
            prior_sd (float, optional): Standard deviation of the prior distribution. Defaults to 1.

        Returns:
            tensor: KL-divergence value.
        """
        
        log_prior = dist.Normal(0, prior_sd).log_prob(z)
        log_p_q = dist.Normal(mu_theta, torch.log(1 + torch.exp(rho_theta))).log_prob(z)
        
        return (log_p_q - log_prior).sum() / z.shape[0] 

    def forward(self, x):
        
        """
        Forward pass of the model.

        Parameters:
            x (tensor): Input data.
            categories (tensor, optional): Input categorical variables. Defaults to None.

        Returns:
            tuple: Tuple containing the latent variable, categorical variable, and posterior probabilities.
        """
        
        # y predictions
        y = x.clone()
        for w_pi, b_pi in zip(self.w_pi[:-1], self.b_pi[:-1]):
            y = self.relu(y.matmul(w_pi) + b_pi)
            
        y = self.softmax(y.matmul(self.w_pi[-1]) + self.b_pi[-1], dim=1)
        
        pi = y.mean(dim=0)
        one_hot_y = torch.eye(x.shape[0], self.nb_cat)[torch.argmax(y, dim=1)]
        
        # z|y predictions
        mu = torch.concat((x, one_hot_y), dim=1)
        for w_mu, b_mu in zip(self.w_mu[:-1], self.b_mu[:-1]):
            mu = self.leakyrelu(mu.matmul(w_mu) + b_mu)
        
        mu = mu.matmul(self.w_mu[-1]) + self.b_mu[-1] # last hidden inference layer of mu, without relu and batch norm

        rho = torch.concat((x, one_hot_y), dim=1)
        for w_rho, b_rho in zip(self.w_rho, self.b_rho):
            rho = self.relu(rho.matmul(w_rho) + b_rho)
            rho = torch.min(rho, torch.tensor(100))
        
        z = self.sampling_z(mu.mean(dim = 0), rho.mean(dim = 0), n_samples = x.shape[0])
        
        # Compute KL-div loss for training
        if self.train_mode:
            self.accumulated_kl_div_y[0] = torch.tensor([self.kl_divergence_y(torch.argmax(y, dim=1), pi, self.prior_pi)], requires_grad=True)
            
            self.accumulated_kl_div_z[0] = self.kl_divergence_z(z, mu.mean(dim=0), rho.mean(dim=0)).clone().detach()

        return z, y, pi
    

class GMVaeGenerationLayer(nn.Module):
    """
    VAE generation block as described in the paper

    Parameters:
        latent_dim (int): The dimensionality of the latent space.
        output_size (int): The size of the output data.
        decoder_hidden_sizes (list, optional): List of hidden layer dimensions for the decoder network. Defaults to [].
        eps (float, optional): A small value to prevent numerical instability. Defaults to 1e-6.
    """

    
    def __init__(self, latent_dim, output_size, decoder_hidden_sizes = [], eps = 1e-6):
        
        super().__init__()
        
        decoder_list = []
        if len(decoder_hidden_sizes)>0:
            a1 = latent_dim
            for hidden_size in decoder_hidden_sizes[:-1]: 
                decoder_list.append(nn.Linear(a1, hidden_size, bias = True))
                a1 = hidden_size   
            decoder_list.append(nn.Linear(a1, output_size))
        else:
            decoder_list.append(nn.Linear(latent_dim, output_size))
            
        self.decoder = nn.ModuleList(decoder_list)
        
        self.relu = nn.ReLU()
        self.eps = eps
    
    def forward(self, z):
        
        """
        Forward pass of the model.

        Parameters:
            z (tensor): Latent variable.

        Returns:
            tensor: Output of the VAE generation block.
        """
        
        out = z.clone()
        for layer in self.decoder:
            out = self.relu(layer(out)) + self.eps
        
        return out

    
class GMVariationalAutoEncoder(nn.Module):
    
    """
    VAE model

    Parameters:
        input_size (int): The size of the input data.
        latent_dim (int): The dimensionality of the latent space.
        nb_cat (int): The number of categories in the categorical distribution.
        prior_pi (tensor, optional): Prior distribution for categorical variables. Defaults to None.
        encoder_hidden_sizes (list, optional): List of hidden layer dimensions for the encoder network. Defaults to [].
        pi_hidden_sizes (list, optional): List of hidden layer dimensions for the categorical distribution network. Defaults to [].
        decoder_hidden_sizes (list, optional): List of hidden layer dimensions for the decoder network. Defaults to [].
        train_mode (bool, optional): Flag indicating whether the model is in training mode. Defaults to True.
    """
    def __init__(self, 
                 input_size,
                 latent_dim,
                 nb_cat,
                 prior_pi = None,
                 encoder_hidden_sizes = [],
                 pi_hidden_sizes = [],
                 decoder_hidden_sizes = [],
                 train_mode = True
                 ):
        
        super().__init__()
        self.accumulated_kl_div_y = torch.zeros(1)
        self.accumulated_kl_div_z = torch.zeros(1)
        self.nb_cat = nb_cat
        self.prior_pi = prior_pi
        self.train_mode = train_mode
        self.encoder = GMVaeInference(input_size,
                                      latent_dim,
                                      self.nb_cat,
                                      self.accumulated_kl_div_y,
                                      self.accumulated_kl_div_z,
                                      self.prior_pi,
                                      encoder_hidden_sizes,
                                      pi_hidden_sizes,
                                      self.train_mode)
        self.decoder = GMVaeGenerationLayer(latent_dim, input_size, decoder_hidden_sizes)

    def reset_kl_div_y(self):
        
        """
        Reset the accumulated KL divergence for categorical variables.
        """

        self.accumulated_kl_div_y[0] = 0.
    
    def reset_kl_div_z(self):
        
        """
        Reset the accumulated KL divergence for latent variables.
        """
        
        for kl in range(len(self.accumulated_kl_div_z)):
            self.accumulated_kl_div_z[kl] = 0.

    def forward(self, x):

        """
        Forward pass of the model.

        Parameters:
            x (tensor): Input data.

        Returns:
            tuple: Tuple containing the categorical variable, output of the VAE generation block, and posterior probabilities.
        """
        
        z, y, pi = self.encoder(x)
        
        out = self.decoder(z)
        
        lambda_ = out.mean(dim=0)
        
        return y, lambda_, pi