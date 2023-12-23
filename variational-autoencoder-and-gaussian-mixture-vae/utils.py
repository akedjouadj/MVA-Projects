import torch
# import torch.distributions as dist
import pyro
import os
import pyro.distributions as dist
import torch.distributions as dist2
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

from torch.utils.tensorboard import SummaryWriter
    
def elbo(input, target, model):
    """
    Compute the Evidence Lower Bound (ELBO) for a given reconstruction distribution.

    Args:
    - input (torch.Tensor or tuple): Input parameters for the reconstruction distribution.
    - target (torch.Tensor): Target values.
    - model: VAE model object containing distribution information and accumulated KL divergence.

    Returns:
    - tuple: Tuple containing log likelihood and negative lower bound values.
    """  
    dist_name= model.dist_name
    if dist_name == 'Poisson':

        reconstructed_dist = dist.Poisson(input)
    elif dist_name == 'NB':
        alpha, beta= input
        reconstructed_dist= dist2.NegativeBinomial(total_count=alpha, probs=beta)
    elif dist_name == 'ZIP':
        psi, lambda_= input
        reconstructed_dist = dist.ZeroInflatedPoisson(rate=psi, gate=lambda_)
    elif dist_name == 'ZINB':
        pi, p, log_r= input
        reconstructed_dist= dist.ZeroInflatedNegativeBinomial(total_count=torch.exp(log_r), probs=p, gate=torch.sigmoid(pi))
    else:
        print('distribution not rcognised')

    log_likelihood = reconstructed_dist.log_prob(target).sum()/target.shape[0]
    lower_bound = log_likelihood - model.accumulated_kl_div[0] 

    return log_likelihood, -lower_bound

def gm_elbo(input, target, model, y=None, categories=None, regularization_method = "kl"):
    
    """
    Compute the ELBO for a reconstruction distribution.

    Parameters:
        input (tensor): Input data for the reconstruction distribution.
        target (tensor): Target data for the reconstruction distribution.
        model (GMVariationalAutoEncoder): The generative model.
        y (tensor, optional): Categorical variable. Defaults to None.
        categories (tensor, optional): Input categories. Defaults to None.
        regularization_method (str, optional): Regularization method to use. 
            Options: "kl" (default), "cross-entropy", "no".
    
    Returns:
        tuple: Tuple containing the terms of the ELBO - (log-likelihood, regularization term, negative lower bound, cross-entropy term).
    """

    log_likelihood_z_y = dist.Poisson(input).log_prob(target).sum()/target.shape[0]
        
    good_prediction_term = log_likelihood_z_y - model.accumulated_kl_div_z
    regularization_term = model.accumulated_kl_div_y[0] 

    crossentropy_y = torch.nn.CrossEntropyLoss()(y, categories.to(torch.long))

    if regularization_method == "kl":
        lower_bound = good_prediction_term -regularization_term
    elif regularization_method == "cross-entropy":
        lower_bound = good_prediction_term + crossentropy_y
    elif regularization_method == "no":
        lower_bound = good_prediction_term
    else:
        raise NotImplementedError
    
    return good_prediction_term, regularization_term, -lower_bound, crossentropy_y

def train_synthetic(model, dataset, lr, weight_decay, nb_epochs, device='cpu'):

    """
    Train the specified model.

    Parameters:
        model (nn.Module): The model to be trained.
        dataset (tensor): Input data for training.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        nb_epochs (int): Number of training epochs.

    Returns:
        Tuple of Lists: ELBO values, log-likelihoods, KL divergence values, and final output.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=weight_decay)

    elbo_values = []
    log_likelihoods = []
    kl_div = []
    dataset= dataset.to(device)
    model = model.to(device)
    print("\nTraining start...")
    for epoch in tqdm(range(nb_epochs)):

        optimizer.zero_grad()
        model.reset_kl_div()

        output = model(dataset)[0]
        
        log_likelihood, loss = elbo(output, dataset, model)
        elbo_values.append(-loss.item())
        log_likelihoods.append(log_likelihood.item())
        kl_div.append(model.accumulated_kl_div[0])
        loss.backward()
        
        optimizer.step()
    
    print(f"\noutput: {output}")
    print(f"\nMeans elbo_value, log_likelihood, kl_div: {np.mean(elbo_values)}, {np.mean(log_likelihoods)}, {np.mean(kl_div)}")
    
    return elbo_values, log_likelihoods, kl_div, output


def train(model, dataloader, lr, weight_decay, nb_epochs, log_path, save_path, dist, device='cpu'):
    """
    Train a VAE model using the specified dataset.

    Args:
    - model: VAE model to be trained.
    - dataloader: DataLoader providing batches of training data.
    - lr (float): Learning rate for the optimizer.
    - weight_decay (float): Weight decay for the optimizer.
    - nb_epochs (int): Number of training epochs.
    - log_path (str): Path to store TensorBoard logs.
    - save_path (str): Path to save the trained model.
    - dist (str): Reconstruction distribution ('Poisson', 'NB', 'ZIP', 'ZINB').
    - device (str): Device for training ('cpu' or 'cuda').

    Returns:
    - tuple: Tuple containing lists of ELBO values, log likelihoods, KL divergence values,
             latent variable values, and descriptors during training.
    """
    writer = SummaryWriter(os.path.join(log_path, 'vae_experiments', '{}'.format(dist)))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elbo_values = []
    log_likelihoods = []
    kl_div_values = []
    z_values= []
    descs=[]

    # print("\nTraining start...")
    count=0
    for epoch in tqdm(range(nb_epochs)):
    #for epoch in range(nb_epochs):
        elbo_=0
        log_likeli=0
        kl_div_=0
        for b_idx, b_data in enumerate(dataloader): #, total=len(dataloader), desc=f"Epoch {epoch+1}")
            b_data = b_data.to(dtype=torch.float32).to(device)
            # print('model device', next(model.parameters()).device)
            # print('data device', b_data.device)
            
            optimizer.zero_grad()
            model.reset_kl_div()
            
            output, z, desc = model(b_data)

            # print('\nthe output of the deocder which will be the inout to elbo is', output[:10])
            if isinstance(output, tuple):
              output = tuple([o.squeeze() for o in output])
            else:
              output = output.squeeze()
            log_likelihood, loss = elbo(output, b_data, model)

            elbo_ +=-loss.item()
            log_likeli += log_likelihood.item()
            kl_div_ +=model.accumulated_kl_div[0] / b_data.shape[0]
            if dist == 'Poisson':
              descs.append(desc)
            z_values.append(z.detach())
            
            
            loss.backward()
            optimizer.step()
        with open(os.path.join(f'/content/drive/MyDrive/pgm_projet/new_scratch/results/lambda_decriptors_max2_{dist}.pickle'), 'wb') as f:
          pickle.dump(descs, f)

        elbo_values.append(elbo_/len(dataloader))
        log_likelihoods.append(log_likeli/len(dataloader))
        kl_div_values.append(kl_div_/len(dataloader))
        # stopping criteria
        if epoch >= 100:
          if abs(elbo_values[epoch]- elbo_values[epoch-1]) < 2:
            count+=1
            if count >5:
              print('\n early stopping at epoch{}'.format(epoch))
              count =0 
              break
          else:
            count=0


        torch.save(model.state_dict(), os.path.join(save_path, 'best_{}.pt'.format(dist)))

        writer.add_scalar('Elbo', np.mean(elbo_values), epoch)
        writer.add_scalar('log_likelihood', np.mean(log_likelihoods), epoch)
        writer.add_scalar('kl_div', np.mean(kl_div_values), epoch)

    writer.close()
    
    return elbo_values, log_likelihoods, kl_div_values, z_values, descs


def test(model, testloader, log_path, save_path, dist, device='cpu'):

    """
    Evaluate a trained VAE model on a test dataset.

    Args:
    - model: Trained VAE model.
    - testloader: DataLoader providing batches of test data.
    - log_path (str): Path to store TensorBoard logs.
    - save_path (str): Path to load the trained model from and save TensorBoard logs.
    - dist (str): Reconstruction distribution ('Poisson', 'NB', 'ZIP', 'ZINB').
    - device (str): Device for testing ('cpu' or 'cuda').

    Returns:
    - tuple: Tuple containing ELBO value and a list of latent variable values during testing.
    """

    writer = SummaryWriter(os.path.join(log_path, 'vae_experiments', '{}'.format(dist)))
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_{}.pt'.format(dist))))
    model.eval()

    z_values= []
    elbo_=0
    with torch.no_grad():
      for b_idx, b_data in enumerate(testloader):
          b_data = b_data.to(dtype=torch.float32).to(device)

          output, z = model(b_data)
          if isinstance(output, tuple):
            output = tuple([o.squeeze() for o in output])
          else:
            output = output.squeeze()
          _, loss = elbo(output, b_data, model)

          elbo_ +=-loss.item()
         
          z_values.append(z.detach())
    elbo_value= elbo_/len(testloader)
    
    writer.add_scalar('Elbo', elbo_value)
    writer.close()
    return elbo_value, z_values


def gm_train(model, dataset, categories, lr, weight_decay, nb_epochs, regularization_method = "kl"):

    """
    Train the Gaussian Mixture Variational Autoencoder (GMVAE) model.

    Parameters:
        model (GMVariationalAutoEncoder): The GMVAE model to be trained.
        dataset (tensor): Input data for training.
        categories (tensor): Input categories for training.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        nb_epochs (int): Number of training epochs.
        regularization_method (str, optional): Regularization method to use during training. 
            Options: "kl" (default), "cross-entropy", "no".

    Returns:
        Tuple of Lists: ELBO values, good prediction terms, regularization terms, cross-entropy losses, 
            final output, and final pi values.
    """
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=weight_decay)

    elbo_values = []
    good_prediction_terms = []
    regularization_terms = []
    crossentropy_losses = []

    print("\nTraining start...")
    for epoch in tqdm(range(nb_epochs)):

        optimizer.zero_grad()
        model.reset_kl_div_y()
        model.reset_kl_div_z()

        outputs = model(dataset)
        y, output, pi = outputs[0], outputs[1].squeeze(), outputs[2]
        
        good_prediction_term, regularization_term, loss, crossentropy_loss = gm_elbo(output, dataset, model, y, categories, regularization_method = "kl")
        elbo_values.append(-loss.item())
        good_prediction_terms.append(good_prediction_term.item())
        regularization_terms.append(regularization_term.item())
        crossentropy_losses.append(crossentropy_loss.item())

        loss.backward(retain_graph=True)
        
        optimizer.step()
    
    print(f"\noutput: {output}")
    print(f"\npi: {pi}")
    print(elbo_values[-1], good_prediction_terms[-1], regularization_terms[-1])
    
    return elbo_values, good_prediction_terms, regularization_terms, crossentropy_losses, output, pi
  






