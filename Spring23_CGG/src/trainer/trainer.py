import torch
import numpy as np
import torch.nn.functional as F
from src.utils.torch_utils import reverse_image_standardisation, compute_psnr
import os, sys, json
import torchvision
from torch import nn
from src.models.models import SimpleModel, SRCNN, LapSRN, CharbonnierLoss, Generator, Discriminator, VGGLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_model_class(config):
    models = {
        'SRCNN': SRCNN(),
        'SimpleModel': SimpleModel(),
        'LapSRN': LapSRN(),
        'SRGAN': Generator(),
        'SRRESNET': Generator()
    }
    return models[config['model']] 


def get_loss_function(config):
    losses = {
        'mse': torch.nn.MSELoss(),
        'charb': CharbonnierLoss()
    }
    return losses[config['loss']]


def get_optimizer(config, model):
    optimizers = {
        'adam': torch.optim.Adam(model.parameters(), lr=config['learning_rate']),
        'adam_beta': torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999))
    }
    return optimizers[config['optimizer']]


def get_metric(config):
    metrics = {
        'psnr': compute_psnr
    }
    return metrics[config['metric_fn']]


def save_model(model, fld_path, epoch, optimizer, loss, model_name):
    model_file_path = get_model_file_path(fld_path, model_name)
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            }, model_file_path)


def get_model_file_path(fld_path, model_name):
    return os.path.join(fld_path, f"{model_name}.pth")


def train_loop(config, train_dataloader, val_dataloader, device,
               rgb_means, rgb_stds, fld_path, trained_model_path=None):
    """ Training loop for network

    Args:
        config (dictionary): config of training parameters
        train_dataloader (Torch Dataloader)
        val_dataloader (Torch Dataloader)
        device (torch.device): 'cpu' or 'cuda'
        rgb_means (list): means from training dataset
        rgb_stds (list): stds from trainign dataset
        fld_path (str): folder path for saving
        trained_model_path (None, str): path to trained model. Defaults to None.
    """

    logging.info(f'start training')

    tensorboard_path = fld_path + "/tensorboard/"
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)

    model = get_model_class(config)
    model.to(device)
    loss_fn = get_loss_function(config)
    optimizer = get_optimizer(config, model)
    num_epochs = config['num_epochs']
    val_metrics = []
    avg_train_losses = []
    best_metric = torch.tensor(1.0e+7)
    start_epoch = 0

    if trained_model_path is not None:
        checkpoint = torch.load(os.path.join(trained_model_path, 'model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f"Continue training from model in {trained_model_path}: starting at epoch {start_epoch+1}")

    num_epochs = num_epochs - start_epoch


    for epoch in range(num_epochs):
        
        epoch = epoch + start_epoch
        logging.info(f'training loop is running: epoch # {epoch+1}')
        train_loss = 0.0
        model.train()
        epoch_loss = []
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)

        for batch, data in loop:
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = model(inputs)

            if model.__class__.__name__ == 'LapSRN':
                loss_x2 = loss_fn(outputs[0], F.interpolate(targets, scale_factor=0.25))
                loss_x4 = loss_fn(outputs[1], F.interpolate(targets, scale_factor=0.5))
                loss_x8 = loss_fn(outputs[2], targets)
                loss = loss_x2 + loss_x4 + loss_x8

            else:
                loss = loss_fn(outputs, targets)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
            # update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=train_loss/(batch+1))

            epoch_loss.append(train_loss)
            train_loss = 0.0

        logging.info(f'Epoch: {epoch + 1}, Epoch loss: {np.mean(epoch_loss):.3f}')

        val_metric = evaluate(val_dataloader, model, loss_fn, device, rgb_means, rgb_stds)
        logging.info(f'Epoch: {epoch + 1}, Validation loss: {val_metric:.3f}')

        writer.add_scalar('training_loss', np.mean(epoch_loss), (epoch+1))
        writer.add_scalar('val_metric', val_metric, (epoch+1))

        val_metrics.append(val_metric)
        avg_train_losses.append(np.mean(epoch_loss))

        # if validation better than last: save
        if (epoch+1) % 5 == 0:
            if best_metric > val_metric:
                best_metric = val_metric
                model_file_path = get_model_file_path(fld_path, 'model')
                save_model(model, fld_path, epoch, optimizer, epoch_loss, 'model')
                logging.info(f'saved model: {model_file_path}')
                logging.info(f'best metric: {best_metric}')

    # save metrics
    for i, metric in enumerate(val_metrics):
        val_metrics[i] = metric.item()

    best_metric = best_metric.item()
    train_metrics = {'train_losses': avg_train_losses,
                     'validation_errors': val_metrics,
                     'best_model_error': best_metric}

    with open(fld_path+'train_metrics.json', 'w') as f:
        json.dump(train_metrics, f)
        
    logging.info(f'Finished Training')


def evaluate(dataloader, model, metric_fn, device, rgb_means, rgb_stds):
    """Computes metric_fn between batch of model outputs and targets

    Args:
        dataloader (torch dataloader)
        model (class): from src.models.models e.g. LapSRN
        metric_fn (class): loss function
        device (torch.device): 'cpu' or 'cuda'
        rgb_means (list): means from training dataset
        rgb_stds (list): stds from trainign dataset

    Returns:
        torch.mean(metric_vals) (torch tensor)
    """
    metric_vals = []
    model.eval()
    with torch.no_grad():
        for img_batch, target_batch in dataloader:
            img_batch = img_batch.to(device)
            target_batch = target_batch.to(device)
            output = model(img_batch)
            if model.__class__.__name__ == 'LapSRN':
                output = output[2]

            metric_val = metric_fn(output, target_batch)
            metric_val = torch.mean(metric_val)
            metric_vals.append(metric_val)

        metric_vals = torch.FloatTensor(metric_vals)

        return torch.mean(metric_vals)


def run_model(dataloader, model, device, rgb_means, rgb_stds):
    """
    - Function is called when testing model and at inference
    - Performs inference and outputs upscaled images with high resolution
    - Saves new images in inference folder
    Args:
        dataloader: containsn inference images in dataloader class (pytorch)
        model: (pytorch class)
        device: CUDA device
        rgb_means: channel means (list of floats)
        rgb_stds: channel stds (list of floats)
    Returns:
        list of images (each image as tensors)
        """

    num_batches = len(dataloader)
    logging.info(f'number of batches: {num_batches}')
    list_of_images = []
    model.eval()
    with torch.no_grad():
        for img_batch in dataloader:
            img_batch = img_batch.to(device)
            output = model(img_batch)

            if model.__class__.__name__ == 'LapSRN':
                output = output[2]
            if rgb_means is not None:
                output = reverse_image_standardisation(output, rgb_means, rgb_stds)

            # append each output to list of inference images
            for i in range(output.size(0)):
                list_of_images.append(output[i, :, :, :])
            logging.info(f'inference on batch done')

        return list_of_images


def model_predict(model_dir, test_dataloader, test_dir, test_filenames, train_config, rgb_means, rgb_stds, device):
    """
    - Function is called when from app.py to test model
    - Performs inference and outputs and saves upscaled images with high resolution
    - Computes model errors and saves results
    """

    test_dir = os.path.basename(os.path.normpath(test_dir))
    output_dir = model_dir + "/inference_" + test_dir + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initialise model
    model = get_model_class(train_config)
    model = model.to(device)
    checkpoint = torch.load(os.path.join(model_dir, "model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])

    inference_images = run_model(test_dataloader, model, device, rgb_means, rgb_stds)

    # save images to testing directory
    for i, image in enumerate(inference_images):
        torchvision.utils.save_image(image, output_dir + test_filenames[i])


def train_loop_gan(config, train_dataloader, val_dataloader, device, rgb_means, rgb_stds, fld_path, trained_model_path):
    """ Training loop for GAN

    Args:
        config (dictionary): config of training parameters
        train_dataloader (Torch Dataloader)
        val_dataloader (Torch Dataloader)
        device (torch.device): 'cpu' or 'cuda'
        rgb_means (list): means from training dataset
        rgb_stds (list): stds from trainign dataset
        fld_path (str): folder path for saving
        trained_model_path (None, str): path to trained model. Defaults to None.
    """
    tensorboard_path = fld_path + "/tensorboard/"
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)

    val_metrics = []
    avg_train_losses = []
    best_metric = 1.0e+7

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    opt_gen = get_optimizer(config, gen)
    opt_disc = get_optimizer(config, disc)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()
    num_epochs = config["num_epochs"]

    if trained_model_path is not None:
        checkpoint = torch.load(os.path.join(trained_model_path, 'model.pth'))
        gen.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        num_epochs = num_epochs - start_epoch
        logging.info(f'Continue training generator from model in {trained_model_path}')

        if os.path.isfile(os.path.join(trained_model_path, 'discriminator.pth')):
            checkpoint = torch.load(os.path.join(trained_model_path, 'discriminator.pth'))
            disc.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f'Continue training discriminator from model in {trained_model_path}'
                         f' starting at epoch {start_epoch+1}')

    else:
        logging.critical("Need to provide SRRESNET Model to start training SRGAN")
        sys.exit

    for epoch in range(num_epochs):
        epoch = epoch + start_epoch
        logging.info(f'training loop is running: epoch # {epoch+1}')
        train_loss = 0.0
        epoch_loss = []
        gen.train()
        disc.train()
        loop = tqdm(train_dataloader, leave=True)

        for batch, (low_res, high_res) in enumerate(loop):
            high_res = high_res.to(device)
            low_res = low_res.to(device)
            
            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            fake = gen(low_res)
            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())
            disc_loss_real = bce(disc_real,
                                 torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real))
            disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = disc_loss_fake + disc_loss_real

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            disc_fake = disc(fake)
            l2_loss = mse(fake, high_res)
            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
            loss_for_vgg = vgg_loss(fake, high_res)
            #gen_loss = l2_loss
            gen_loss = l2_loss + loss_for_vgg + adversarial_loss

            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            train_loss += gen_loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=train_loss/(batch+1))
            epoch_loss.append(train_loss)
            train_loss = 0.0
        
        logging.info(f'Epoch: {epoch + 1}, Epoch loss: {np.mean(epoch_loss):.3f}')

        val_metric = evaluate(val_dataloader, gen, mse, device, rgb_means, rgb_stds)
        logging.info(f'Epoch: {epoch + 1}, Validation loss: {val_metric:.3f}')

        writer.add_scalar('training_loss', np.mean(epoch_loss), (epoch+1))
        writer.add_scalar('val_metric', val_metric, (epoch+1))

        val_metrics.append(val_metric)
        avg_train_losses.append(np.mean(epoch_loss))

        # if validation better than last: save
        if (epoch+1) % 1 == 0:
            if best_metric > val_metric:
                best_metric = val_metric
                model_file_path = get_model_file_path(fld_path, 'model')
                save_model(gen, fld_path, epoch, opt_gen, epoch_loss, model_name='model')
                save_model(disc, fld_path, epoch, opt_disc, epoch_loss, model_name='discriminator')
                logging.info(f'saved model: {model_file_path}')
                logging.info(f'best metric: {best_metric}')

    logging.info(f'saving metrics')
    for i, metric in enumerate(val_metrics):
        val_metrics[i] = metric.item()
    best_metric = best_metric.item()
    train_metrics = {'train_losses': avg_train_losses,
                     'validation_errors': val_metrics,
                     'best_model_error': best_metric}
    with open(fld_path+'train_metrics.json', 'w') as f:
        json.dump(train_metrics, f)
    logging.info(f'Finished Training')

