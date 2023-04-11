import os, argparse, configparser, logging, pickle, json
import src.utils.np_utils as npu
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.data.datasets import SRDataset
from src.trainer.trainer import train_loop, train_loop_gan, model_predict
from src.utils.torch_utils import compose_transforms_dict
from src.utils.np_utils import create_patches
from src.test import compute_mse_errors, compute_psnr_errors, compute_ssim_errors
import torch
from datetime import datetime

logging.basicConfig(
    filename='log_file_name.log',
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


"""
Usage:
    python app.py --mode [options] --config [path]

Options:
    --mode      run training, testing or inference
    --config    path to configuration file

Config file:
[TRAINING]
low_res_dir                         path to training folder (low-res images)
ground_truth_dir                    path to training folder (high-res images)
random_seed                         random state for shuffling dataset (int)
train_split                         fraction of images to be allocated to train (e.g. 0.8)
flip_horizontal                     augment data by randomly performing random horizontal flips (True or False)
flip_vertical                       augment data by randomly performing random vertical flips (True or False)
pad_training                        if padding, give size to pad low-res images to (channel, height, width): e.g. [3, 150, 150]
pad_target                          if padding, give size to pad ground truth images to (channel, height, width): e.g. [3, 1000, 1000]
model                               model name (e.g. lapsrn)
optimiser                           optimiser name (adam or sgd)
learning_rate                       learning rate for optimiser (float)
num_epochs                          number of epochs
loss                                loss function name (mse or charb)     
batch_size                          size of batch
standardisation                     standardises training data if True, normalises training data if False (True or False)

[INFERENCE]
model_dir                           path to folder containing a trained model.pth file
low_res_dir                         path to folder containing test images (low-res)

[EVALUATION]
model_dir                           path to folder containing a trained model.pth file
"""


def create_output_folder(model_name):
    time_now = datetime.now().replace(second=0, microsecond=0).strftime("%d_%m_%Y_%H%M")
    filename = '{}_{}'.format(model_name, time_now)
    fld_path = "models/" + filename + "/"
    if not os.path.exists(fld_path):
        os.makedirs(fld_path)
    return fld_path


def train(config):
    logging.info("Start training")

    try:
        config_dictionary = dict(config.items('TRAINING'))
        config_dictionary["low_res_dir"] = config_dictionary["low_res_dir"] + "/"
        config_dictionary["ground_truth_dir"] = config_dictionary["ground_truth_dir"] + "/"
        config_dictionary["random_seed"] = int(config_dictionary["random_seed"])
        config_dictionary["train_split"] = float(config_dictionary["train_split"])
        config_dictionary["standardisation"] = False if config_dictionary["standardisation"] == 'false' else True
        config_dictionary["flip_horizontal"] = False if config_dictionary["flip_horizontal"] == 'false' else True
        config_dictionary["flip_vertical"] = False if config_dictionary["flip_vertical"] == 'false' else True
        config_dictionary["learning_rate"] = float(config_dictionary["learning_rate"])
        config_dictionary["num_epochs"] = int(config_dictionary["num_epochs"])
        config_dictionary["batch_size"] = int(config_dictionary["batch_size"])
    except:
        logging.critical(f"Something is wrong with your config file- make sure it's written properly")
        exit()
    logging.info("Extracted config file")

    if "pad_training" in config_dictionary:
        config_dictionary["pad_training"] = eval(config_dictionary["pad_training"])
    if "pad_target" in config_dictionary:
        config_dictionary["pad_target"] = eval(config_dictionary["pad_target"])

    if "trained_model_path" in config_dictionary:
        trained_model_path = config_dictionary["trained_model_path"]
    else:
        trained_model_path = None

    # create main model directory
    fld_path = create_output_folder(config_dictionary["model"])

    # dump config params used in this training run
    with open(fld_path+'training_config.json', 'w') as f:
        json.dump(config_dictionary, f)

    # Only for SRCNN
    if config_dictionary["model"] == "SRCNN" or\
            config_dictionary["model"] == "SRGAN" or\
            config_dictionary["model"] == "SRRESNET":

        if config_dictionary["create_patches"] == "true":
            create_patches(config_dictionary["model"],
                           config_dictionary["ground_truth_dir"],
                           config_dictionary["high_res_patches_path"]+"/",
                           config_dictionary["low_res_patches_path"]+"/",
                           int(config_dictionary["lr_patches_down_scale"]),
                           SIZE=300,
                           STRIDE=250)

        config_dictionary["low_res_dir"] = config_dictionary["low_res_patches_path"]+"/"
        config_dictionary["ground_truth_dir"] = config_dictionary["high_res_patches_path"]+"/"

    # load list of filenames for low-res and GT
    logging.info(f'load low-res filenames from: {config_dictionary["low_res_dir"]}')
    logging.info(f'load ground truth filenames from: {config_dictionary["ground_truth_dir"]}')
    try:
        lr_filenames = os.listdir(config_dictionary["low_res_dir"])
    except FileNotFoundError:
        logging.critical(f'No files in low-res folder - {config_dictionary["low_res_dir"]} !!!')
        exit()
    try:
        hr_filenames = os.listdir(config_dictionary["ground_truth_dir"])
    except FileNotFoundError:
        logging.critical(f'No files in high-res folder!!! ({config_dictionary["ground_truth_dir"]})')
        exit()

    ## make sure number of images in low-res and high-res is equal
    if len(lr_filenames)!= len(hr_filenames):
        logging.critical(f'Number of images in low-resolution directory ({len(lr_filenames)})'
                         f' is not the same as in high-resolution ({len(hr_filenames)})')
        exit()
    ## make sure names of images in low-res and high-res are equal
    if not set(lr_filenames) == set(hr_filenames):
        img_names_difference = set(hr_filenames) - set(lr_filenames)
        img_names_difference.update(set(lr_filenames) - set(hr_filenames))
        logging.critical(f'Names of images in low-resolution directory '
                         f' are not the same as in high-resolution. missing file names: {img_names_difference}')
        exit()

    # determine training, validation split and save filenames in JSON
    logging.info(f'Splitting data to train, validation')
    low_res_filenames_train, low_res_filenames_validation, high_res_filenames_train, high_res_filenames_validation =\
        train_test_split(
            lr_filenames,
            hr_filenames,
            train_size=config_dictionary["train_split"],
            random_state=config_dictionary["random_seed"])

    train_val_split = {'train_filenames': low_res_filenames_train, 'validation_filenames': low_res_filenames_validation}

    with open(fld_path+'train_val_split.json', 'w') as f:
        json.dump(train_val_split, f)

    if config_dictionary["standardisation"]:
        logging.info(f'compute means, std of training set')

        # compute means and std of training set
        rgb_means, rgb_stds = npu.compute_stats_channel_dim(config_dictionary["low_res_dir"], low_res_filenames_train)
        standardisation_stats = {'means': rgb_means, 'stds': rgb_stds}
        with open(fld_path+'standardisation_rgb_stats.json', 'w') as f:
            json.dump(standardisation_stats, f)

    else:
        rgb_means, rgb_stds = None, None

    # compose the dictionary of pytorch transformations
    logging.info(f'compose data transforms and datasets')
    data_transforms = compose_transforms_dict(config_dictionary,
                                              rgb_means,
                                              rgb_stds)

    # compose pytorch dataset classes
    train_dataset = SRDataset(
        fnames=low_res_filenames_train,
        img_dir=config_dictionary["low_res_dir"],
        target_dir=config_dictionary["ground_truth_dir"],
        transform=data_transforms['train_input'],
        target_transform=data_transforms['train_target']
    )
    val_dataset = SRDataset(
        fnames=low_res_filenames_validation,
        img_dir=config_dictionary["low_res_dir"],
        target_dir=config_dictionary["ground_truth_dir"],
        transform=data_transforms['val_input'],
        target_transform=data_transforms['val_target']
    )

    # compose pytorch dataloader classes
    logging.info(f'compose pytorch dataloaders')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config_dictionary["batch_size"],
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config_dictionary["batch_size"],
                                shuffle=False)

    # run training loop
    logging.info(f'run training loop for model {config_dictionary["model"]}, with '
                 f'{int(config_dictionary["num_epochs"])} epochs')

    if config_dictionary["model"] == "SRGAN":
        train_loop_gan(config_dictionary,
                       train_dataloader,
                       val_dataloader,
                       device,
                       rgb_means,
                       rgb_stds,
                       fld_path,
                       trained_model_path)

    else:
        train_loop(config_dictionary,
                   train_dataloader,
                   val_dataloader,
                   device,
                   rgb_means,
                   rgb_stds,
                   fld_path,
                   trained_model_path)


def inference(config):
    config_dictionary = dict(config.items('INFERENCE'))
    config_dictionary["model_dir"] = config_dictionary["model_dir"] + "/"

    # load necessary params for inference
    train_config_file = open(config_dictionary["model_dir"] + "training_config.json")
    training_config_dictionary = json.load(train_config_file)
    if os.path.exists(config_dictionary["model_dir"] + "standardisation_rgb_stats.json"):
        standardisation_file = open(config_dictionary["model_dir"] + "standardisation_rgb_stats.json")
        standardisation = json.load(standardisation_file)
        rgb_means = standardisation["means"]
        rgb_stds = standardisation["stds"]
    else:
        rgb_means = None
        rgb_stds = None

    # if low_res_dir is passed, perform inference on given directory
    if "low_res_dir" in config_dictionary:
        low_res_dir = config_dictionary["low_res_dir"] + "/"

        logging.info(f'load low-res filenames from: {low_res_dir}')
        filenames_inference = os.listdir(low_res_dir)

        # compose the data transform, test dataset and test dataloader
        logging.info(f'compose data transforms, test dataset and dataloader')
        data_transforms = compose_transforms_dict(training_config_dictionary, rgb_means, rgb_stds)
        test_dataset = SRDataset(
            fnames=filenames_inference,
            img_dir=config_dictionary["low_res_dir"],
            target_dir=None,
            transform=data_transforms['test_input'],
            target_transform=None
        )
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=training_config_dictionary["batch_size"],
                                     shuffle=False)

        # perform inference
        logging.info(f'testing model on unseen data')
        model_predict(config_dictionary["model_dir"], test_dataloader, low_res_dir, filenames_inference, training_config_dictionary, rgb_means, rgb_stds, device)

    # if no low_res_dir, perform inference on validation set (used for training)
    else:
        low_res_dir = training_config_dictionary["low_res_dir"]

        # get list of filenames used in validation
        logging.info(f'loading validation dataset filenames')
        train_val_split_file = open(config_dictionary["model_dir"] + "train_val_split.json")
        train_val_split = json.load(train_val_split_file)
        low_res_filenames_validation = train_val_split["validation_filenames"]

        # compose the data transform, test dataset and test dataloader
        logging.info(f'compose data transforms, test dataset and dataloader')
        data_transforms = compose_transforms_dict(training_config_dictionary, rgb_means, rgb_stds)
        test_dataset = SRDataset(
            fnames=low_res_filenames_validation,
            img_dir=training_config_dictionary["low_res_dir"],
            target_dir=None,
            transform=data_transforms['test_input'],
            target_transform=None)
        
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=training_config_dictionary["batch_size"],
                                     shuffle=False)

        # perform inference
        logging.info(f'testing model on validation dataset')
        model_predict(config_dictionary["model_dir"],
                      test_dataloader,
                      training_config_dictionary["low_res_dir"],
                      low_res_filenames_validation,
                      training_config_dictionary,
                      rgb_means,
                      rgb_stds,
                      device)


def evaluate(config):
    logging.info("Start model performance evaluation")
    config_dictionary = dict(config.items('EVALUATE'))
    config_dictionary["model_dir"] = config_dictionary["model_dir"] + "/"

    # if low_res_dir is passed, no gt (only no reference metrics)
    if "low_res_dir" in config_dictionary:
        pass

    # if not, do comparison on validation test split
    else:
        logging.info(f"evaluating model by comparing inference vs gt on validation set")
        train_config_file = open(config_dictionary["model_dir"] + "training_config.json")
        training_config_dictionary = json.load(train_config_file)
        train_val_split_file = open(config_dictionary["model_dir"] + "train_val_split.json")
        train_val_split = json.load(train_val_split_file)

        prediction_directory = config_dictionary["model_dir"] + 'inference_' +\
                               os.path.basename(os.path.normpath(training_config_dictionary["low_res_dir"])) + "/"
        
        if not os.path.exists(prediction_directory):
            logging.critical(f"Aborting. Missing inference folder with validation images: " + prediction_directory)
            exit()
            
        ground_truth_directory = training_config_dictionary["ground_truth_dir"]
        validation_filenames = train_val_split["validation_filenames"]

        # load list of filenames for low-res
        logging.info(f'load predictions from: {prediction_directory}')
        logging.info(f'load ground truth images from: {ground_truth_directory}')
        prediction_images = []
        ground_truth_images = []
        for filename in validation_filenames:
            prediction_images.append(npu.load_image_as_np(prediction_directory + filename))
            ground_truth_images.append(npu.load_image_as_np(ground_truth_directory + filename))

        # compute model performance stats
        logging.info(f'compute mse errors')
        mse_dict = compute_mse_errors(ground_truth_images, prediction_images, validation_filenames)
        logging.info(f'compute psnr errors')
        psnr_dict = compute_psnr_errors(ground_truth_images, prediction_images, validation_filenames)
        logging.info(f'compute ssim errors')
        ssim_dict = compute_ssim_errors(ground_truth_images, prediction_images, validation_filenames)

        # compile into JSON file and save in model_dir
        logging.info(f'saving errors json: ' + config_dictionary["model_dir"] + "errors_validation_set.json")
        errors_dict = {**mse_dict, **psnr_dict, **ssim_dict}
        with open(config_dictionary["model_dir"] + "errors_validation_set.json", 'w') as f:
            json.dump(errors_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', '-m',
        help='Set mode: train, inference or evaluate.',
        default='train',
        type=str,
        choices=['train', 'inference', 'evaluate'],
    )
    parser.add_argument(
        '--config', '-c',
        help='Set the full path to the config file.',
        type=str,
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    if args.mode == "train":
        train(config)
    if args.mode == "inference":
        inference(config)
    if args.mode == "evaluate":
        evaluate(config)