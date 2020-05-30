import os
import argparse
import time

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models.definitions.perceptual_loss_net import PerceptualLossNet
from models.definitions.transformer_net import TransformerNet
import utils.utils as utils


def train(training_config):
    writer = SummaryWriter()  # (tensorboard) writer will output to ./runs/ directory by default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data loader
    train_loader = utils.get_training_data_loader(training_config)

    # prepare neural networks
    transformer_net = TransformerNet().train().to(device)
    perceptual_loss_net = PerceptualLossNet(requires_grad=False).to(device)

    optimizer = Adam(transformer_net.parameters())

    # Calculate style image's Gram matrices (style representation)
    # Built over feature maps as produced by the perceptual net - VGG16
    style_img_path = os.path.join(training_config['style_images_path'], training_config['style_img_name'])
    style_img = utils.prepare_img(style_img_path, target_shape=None, device=device, batch_size=training_config['batch_size'])
    style_img_set_of_feature_maps = perceptual_loss_net(style_img)
    target_style_representation = [utils.gram_matrix(x) for x in style_img_set_of_feature_maps]

    utils.print_header(training_config)
    # Tracking loss metrics, NST is ill-posed we can only track loss and visual appearance of the stylized images
    acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]
    ts = time.time()
    for epoch in range(training_config['num_of_epochs']):
        for batch_id, (content_batch, _) in enumerate(train_loader):
            # step1: Feed content batch through transformer net
            content_batch = content_batch.to(device)
            stylized_batch = transformer_net(content_batch)

            # step2: Feed content and stylized batch through perceptual net (VGG16)
            content_batch_set_of_feature_maps = perceptual_loss_net(content_batch)
            stylized_batch_set_of_feature_maps = perceptual_loss_net(stylized_batch)

            # step3: Calculate content representations and content loss
            target_content_representation = content_batch_set_of_feature_maps.relu2_2
            current_content_representation = stylized_batch_set_of_feature_maps.relu2_2
            content_loss = training_config['content_weight'] * torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

            # step4: Calculate style representation and style loss
            style_loss = 0.0
            current_style_representation = [utils.gram_matrix(x) for x in stylized_batch_set_of_feature_maps]
            for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
                style_loss += torch.nn.MSELoss(reduction='mean')(gram_gt, gram_hat)
            style_loss /= len(target_style_representation)
            style_loss *= training_config['style_weight']

            # step5: Calculate total variation loss - enforces image smoothness
            tv_loss = training_config['tv_weight'] * utils.total_variation(stylized_batch)

            # step6: Combine losses and do a backprop
            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()

            optimizer.zero_grad()  # clear gradients for the next round

            #
            # Logging and checkpoint creation
            #
            acc_content_loss += content_loss.item()
            acc_style_loss += style_loss.item()
            acc_tv_loss += tv_loss.item()

            if training_config['enable_tensorboard']:
                # log scalars
                writer.add_scalar('Loss/content-loss', content_loss.item(), len(train_loader) * epoch + batch_id + 1)
                writer.add_scalar('Loss/style-loss', style_loss.item(), len(train_loader) * epoch + batch_id + 1)
                writer.add_scalar('Loss/tv-loss', tv_loss.item(), len(train_loader) * epoch + batch_id + 1)
                writer.add_scalars('Statistics/min-max-mean-median', {'min': torch.min(stylized_batch), 'max': torch.max(stylized_batch), 'mean': torch.mean(stylized_batch), 'median': torch.median(stylized_batch)}, len(train_loader) * epoch + batch_id + 1)
                # log stylized image
                if batch_id % training_config['image_log_freq'] == 0:
                    stylized = utils.post_process_image(stylized_batch[0].detach().to('cpu').numpy())
                    stylized = np.moveaxis(stylized, 2, 0)  # writer expects channel first image
                    writer.add_image('stylized_img', stylized, len(train_loader) * epoch + batch_id + 1)

            if training_config['console_log_freq'] is not None and batch_id % training_config['console_log_freq'] == 0:
                print(f'time elapsed={(time.time()-ts)/60:.2f}[min]|epoch={epoch + 1}|batch=[{batch_id + 1}/{len(train_loader)}]|c-loss={acc_content_loss / training_config["console_log_freq"]}|s-loss={acc_style_loss / training_config["console_log_freq"]}|tv-loss={acc_tv_loss / training_config["console_log_freq"]}|total loss={(acc_content_loss + acc_style_loss + acc_tv_loss) / training_config["console_log_freq"]}')
                acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]

            if training_config['checkpoint_freq'] is not None and (batch_id + 1) % training_config['checkpoint_freq'] == 0:
                training_state = utils.get_training_metadata(training_config)
                training_state["state_dict"] = transformer_net.state_dict()
                training_state["optimizer_state"] = optimizer.state_dict()
                ckpt_model_name = f"ckpt_style_{training_config['style_img_name'].split('.')[0]}_cw_{str(training_config['content_weight'])}_sw_{str(training_config['style_weight'])}_tw_{str(training_config['tv_weight'])}_epoch_{epoch}_batch_{batch_id}.pth"
                torch.save(training_state, os.path.join(training_config['checkpoints_path'], ckpt_model_name))

    #
    # Save model with additional metadata - like which commit was used to train the model, style/content weights, etc.
    #
    training_state = utils.get_training_metadata(training_config)
    training_state["state_dict"] = transformer_net.state_dict()
    training_state["optimizer_state"] = optimizer.state_dict()
    model_name = f"style_{training_config['style_img_name'].split('.')[0]}_datapoints_{training_state['num_of_datapoints']}_cw_{str(training_config['content_weight'])}_sw_{str(training_config['style_weight'])}_tw_{str(training_config['tv_weight'])}.pth"
    torch.save(training_state, os.path.join(training_config['model_binaries_path'], model_name))


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'mscoco')
    style_images_path = os.path.join(os.path.dirname(__file__), 'data', 'style-images')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    checkpoints_root_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
    image_size = 256  # training images from MS COCO are resized to image_size x image_size
    batch_size = 4

    assert os.path.exists(dataset_path), f'MS COCO missing. Download the dataset using resource_downloader.py script.'
    os.makedirs(model_binaries_path, exist_ok=True)

    #
    # Modifiable args - feel free to play with these (only a small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    # training related
    parser.add_argument("--style_img_name", type=str, help="style image name that will be used for training", default='edtaonisl.jpg')
    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e0)  # you don't need to change this one just play with style loss
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=4e5)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=0)
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs ", default=2)
    parser.add_argument("--subset_size", type=int, help="number of MS COCO images (NOT BATCHES) to use, default is all (~83k)(specified by None)", default=None)
    # logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging (scalars + images)", default=True)
    parser.add_argument("--image_log_freq", type=int, help="tensorboard image logging (batch) frequency - enable_tensorboard must be True to use", default=100)
    parser.add_argument("--console_log_freq", type=int, help="logging to output console (batch) frequency", default=500)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (batch) frequency", default=2000)
    args = parser.parse_args()

    checkpoints_path = os.path.join(checkpoints_root_path, args.style_img_name.split('.')[0])
    if args.checkpoint_freq is not None:
        os.makedirs(checkpoints_path, exist_ok=True)

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['dataset_path'] = dataset_path
    training_config['style_images_path'] = style_images_path
    training_config['model_binaries_path'] = model_binaries_path
    training_config['checkpoints_path'] = checkpoints_path
    training_config['image_size'] = image_size
    training_config['batch_size'] = batch_size

    # Original J.Johnson's training with improved transformer net architecture
    train(training_config)

