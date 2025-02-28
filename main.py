import os
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from load_data import load_CelebAHQ256
from util import training_loss, sampling, sampling_ddim
from util import rescale, find_max_epoch, print_size
from util import linear_beta_schedule, quadratic_beta_schedule
from util import cosine_beta_schedule, sigmoid_beta_schedule

from UNet import UNet

device = "cuda:1"

def train(output_directory, ckpt_epoch, n_epochs, learning_rate, batch_size, 
          T, beta_0, beta_T, unet_config):
    """
    Train the UNet model on the CELEBA-HQ 256 * 256 dataset

    Parameters:

    output_directory (str):     save model checkpoints to this path
    ckpt_epoch (int or 'max'):  the pretrained model checkpoint to be loaded; 
                                automitically selects the maximum epoch if 'max' is selected
    n_epochs (int):             number of epochs to train
    learning_rate (float):      learning rate
    batch_size (int):           batch size
    T (int):                    the number of diffusion steps
    beta_0 and beta_T (float):  diffusion parameters
    unet_config (dict):         dictionary of UNet parameters
    """

    # Compute diffusion hyperparameters
    Beta = torch.linspace(beta_0, beta_T, T).to(device)
    Alpha = 1 - Beta
    Alpha_bar = torch.ones(T).to(device)
    Beta_tilde = Beta + 0
    for t in range(T):
        Alpha_bar[t] *= Alpha[t] * Alpha_bar[t-1] if t else Alpha[t]
        if t > 0:
            Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)

    # Load training data
    trainloader = load_CelebAHQ256(batch_size=batch_size)
    print('Data loaded')

    # Predefine model
    net = UNet(**unet_config).to(device)
    print_size(net)

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Load checkpoint
    time0 = time.time()
    if ckpt_epoch == 'max':
        ckpt_epoch = find_max_epoch(output_directory, 'unet_ckpt')
    if ckpt_epoch >= 0:
        model_path = os.path.join(output_directory, 'unet_ckpt_' + str(ckpt_epoch) + '.pkl')
        checkpoint = torch.load(model_path, map_location='cpu')
        print('Model at epoch %s has been trained for %s seconds' % (ckpt_epoch, checkpoint['training_time_seconds']))
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        time0 -= checkpoint['training_time_seconds']
        print('checkpoint model loaded successfully')
    else:
        ckpt_epoch = -1
        print('No valid checkpoint model found, start training from initialization.')

    # Start training
    for epoch in range(ckpt_epoch + 1, n_epochs):
        for i, (X, _) in enumerate(trainloader):
            X = X.to(device)
            
            # Back-propagation
            optimizer.zero_grad()
            loss = training_loss(net, nn.MSELoss(), T, X, Alpha_bar)
            loss.backward()
            optimizer.step()
            
            # Print training loss
            if i % 100 == 0:
                print("epoch: {}, iter: {}, loss: {:.7f}".format(epoch, i, loss), flush=True)

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_time_seconds': int(time.time()-time0)}, 
                        os.path.join(output_directory, 'unet_ckpt_' + str(epoch) + '.pkl'))
            print('model at epoch %s is saved' % epoch)


def generate(output_directory, ckpt_path, ckpt_epoch, n,
             T, beta_0, beta_T, unet_config):
    """
    Generate images using the pretrained UNet model

    Parameters:

    output_directory (str):     output generated images to this path
    ckpt_path (str):            path of the checkpoints
    ckpt_epoch (int or 'max'):  the pretrained model checkpoint to be loaded; 
                                automitically selects the maximum epoch if 'max' is selected
    n (int):                    number of images to generate
    T (int):                    the number of diffusion steps
    beta_0 and beta_T (float):  diffusion parameters
    unet_config (dict):         dictionary of UNet parameters
    """

    # Compute diffusion hyperparameters
    Beta = cosine_beta_schedule(T).to(device)
    Beta = linear_beta_schedule(T, beta_0, beta_T).to(device)
    Alpha = 1 - Beta
    Alpha_bar = torch.ones(T).to(device)
    Beta_tilde = Beta + 0
    for t in range(T):
        Alpha_bar[t] *= Alpha[t] * Alpha_bar[t-1] if t else Alpha[t]
        if t > 0:
            Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)

    # Predefine model
    net = UNet(**unet_config).to(device)
    print(net)
    print_size(net)

    # Load checkpoint
    if ckpt_epoch == 'max':
        ckpt_epoch = find_max_epoch(ckpt_path, 'unet_ckpt')
    model_path = os.path.join(ckpt_path, 'unet_ckpt_' + str(ckpt_epoch) + '.pkl')
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print('Model at epoch %s has been trained for %s seconds' % (ckpt_epoch, checkpoint['training_time_seconds']))
        net = UNet(**unet_config)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(device)
    except:
        raise Exception('No valid model found')

    # Generation
    time0 = time.time()
    #X_gen = sampling(net, (n,3,256,256), T, Alpha, Alpha_bar, Sigma)
    X_gen = sampling_ddim(net, (n,3,256,256), T, Alpha, Alpha_bar, output_directory, eta=0.3)
    print('generated %s samples at epoch %s in %s seconds' % (n, ckpt_epoch, int(time.time()-time0)))

    # Save generated images
    for i in range(n):
        save_image(rescale(X_gen[i]), os.path.join(output_directory, 'img2_{}.jpg'.format(i)))
    print('saved generated samples at epoch %s' % ckpt_epoch)

def change_to_current_folder():
    # Obtenir le chemin du fichier en cours d'exécution
    current_file_path = os.path.abspath(__file__)
    # Obtenir le répertoire parent
    parent_dir = os.path.dirname(current_file_path)
    # Se placer dans le répertoire parent
    os.chdir(parent_dir)
    print("Répertoire courant : ", os.getcwd())


if __name__ == "__main__":
    change_to_current_folder()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-t', '--task', type=str, choices=['train', 'generate'],
                        help='Run either training or generation')
    args = parser.parse_args()

    # parse configs
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    unet_config         = config["unet_config"]
    diffusion_config    = config["diffusion_config"]
    train_config        = config["train_config"]
    sample_config          = config["gen_config"]

    # Creates all intermediate folders if necessary
    os.makedirs(train_config["output_directory"], exist_ok=True)
    os.makedirs(sample_config["output_directory"], exist_ok=True)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # go to task
    if args.task == 'train':
        train(**train_config, **diffusion_config, unet_config=unet_config)
    elif args.task =='generate':
        generate(**sample_config, **diffusion_config, unet_config=unet_config)
    else:
        raise Exception("Task is not valid.")
