import os
import argparse
import time

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import git

from models.definitions.perceptual_loss_net import PerceptualLossNet
from models.definitions.transformer_net import TransformerNet
import utils.utils as utils


def train(training_config):
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = utils.get_training_data_loader(training_config)

    transformer_net = TransformerNet().train().to(device)
    perceptual_loss_net = PerceptualLossNet(requires_grad=False).to(device)

    optimizer = Adam(transformer_net.parameters())

    # Calculate style image's Gram matrices (style representation)
    # Built over feature maps as produced by the perceptual net - VGG16
    style_img_path = os.path.join(training_config['style_images_path'], training_config['style_img_name'])
    style_img = utils.prepare_img(style_img_path, target_shape=None, device=device, repeat=training_config['batch_size'])
    style_img_set_of_feature_maps = perceptual_loss_net(style_img)
    target_style_representation = [utils.gram_matrix(x) for x in style_img_set_of_feature_maps]

    utils.print_header(training_config)
    # Tracking loss metrics, NST is ill-posed we can only track loss and visual appearance of the stylized images
    acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]
    ts = time.time()
    for epoch in range(training_config['num_of_epochs']):
        for batch_id, (content_batch, _) in enumerate(train_loader):
            # We want to train the transformer net to work on [0, 255] imagery (no ImageNet mean normalization here)
            content_batch = content_batch.to(device)
            stylized_batch = transformer_net(content_batch)

            # stylized_batch = utils.special_preprocessing(stylized_batch)

            # We need to normalize these because the perceptual loss net is VGG16 and it was trained on normalized imgs
            # normalized_content_batch = utils.normalize_batch(content_batch)
            # normalized_stylized_batch = utils.normalize_batch(stylized_batch)
            # print(torch.max(normalized_content_batch), torch.min(normalized_content_batch))
            # print(torch.max(normalized_stylized_batch), torch.min(normalized_stylized_batch))

            # Calculate content representations
            content_batch_set_of_feature_maps = perceptual_loss_net(content_batch)
            stylized_batch_set_of_feature_maps = perceptual_loss_net(stylized_batch)
            target_content_representation = content_batch_set_of_feature_maps.relu2_2
            current_content_representation = stylized_batch_set_of_feature_maps.relu2_2

            content_loss = training_config['content_weight'] * torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

            style_loss = 0.0
            current_style_representation = [utils.gram_matrix(x) for x in stylized_batch_set_of_feature_maps]
            for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
                style_loss += torch.nn.MSELoss(reduction='mean')(gram_gt, gram_hat)
            style_loss /= len(target_style_representation)
            style_loss *= training_config['style_weight']

            tv_loss = training_config['tv_weight'] * utils.total_variation(stylized_batch)  # enforces image smoothness

            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #
            # Logging and checkpoint creation
            #
            acc_content_loss += content_loss.item()
            acc_style_loss += style_loss.item()
            acc_tv_loss += tv_loss.item()

            # todo: add tensorboard to env file and enable tensorboard setting
            writer.add_scalar('Loss/content-loss', content_loss.item(), len(train_loader) * epoch + batch_id + 1)
            writer.add_scalar('Loss/style-loss', style_loss.item(), len(train_loader) * epoch + batch_id + 1)
            writer.add_scalar('Loss/tv-loss', tv_loss.item(), len(train_loader) * epoch + batch_id + 1)
            writer.add_scalars('Statistics/min-max-mean-median', {'min': torch.min(stylized_batch), 'max': torch.max(stylized_batch), 'mean': torch.mean(stylized_batch), 'median': torch.median(stylized_batch)}, len(train_loader) * epoch + batch_id + 1)

            if training_config['log_freq'] is not None and batch_id % training_config['log_freq'] == 0:
                print(f'time elapsed={(time.time()-ts)/60:.2f}[min]|epoch={epoch + 1}|batch=[{batch_id + 1}/{len(train_loader)}]|c-loss={acc_content_loss / training_config["log_freq"]}|s-loss={acc_style_loss / training_config["log_freq"]}|tv-loss={acc_tv_loss / training_config["log_freq"]}|total loss={(acc_content_loss + acc_style_loss + acc_tv_loss) / training_config["log_freq"]}')
                acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]
                with torch.no_grad():  # todo: consider using detach() it's more concise
                    import numpy as np
                    from PIL import Image
                    tmp = utils.post_process(stylized_batch[0].to('cpu').numpy())
                    print(tmp.dtype, tmp.shape, np.max(tmp))
                    tmp = np.moveaxis(tmp, 2, 0)
                    writer.add_image('stylized_img', tmp, len(train_loader) * epoch + batch_id + 1)

            if training_config['checkpoint_freq'] is not None and (batch_id + 1) % training_config['checkpoint_freq'] == 0:
                ckpt_model_name = f"ckpt_style_{training_config['style_img_name'].split('.')[0]}_cw_{str(training_config['content_weight'])}_sw_{str(training_config['style_weight'])}_tw_{str(training_config['tv_weight'])}_epoch_{epoch}_batch_{batch_id}.pth"
                torch.save(transformer_net.state_dict(), os.path.join(training_config['checkpoints_path'], ckpt_model_name))

    #
    # Save model with additional metadata - like which commit was used to train the model, style/content weights, etc.
    #
    num_of_datapoints = training_config['subset_size'] * training_config['num_of_epochs']
    state = {
        "state_dict": transformer_net.state_dict(),
        "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "content_weight": training_config['content_weight'],
        "style_weight": training_config['style_weight'],
        "num_of_datapoints": num_of_datapoints
    }

    model_name = f"style_{training_config['style_img_name'].split('.')[0]}_datapoints_{num_of_datapoints}_cw_{str(training_config['content_weight'])}_sw_{str(training_config['style_weight'])}_tw_{str(training_config['tv_weight'])}.pth"
    torch.save(state, os.path.join(training_config['model_binaries_path'], model_name))


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'mscoco')
    style_images_path = os.path.join(os.path.dirname(__file__), 'data', 'style-images')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    checkpoints_root_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
    image_size = 256
    batch_size = 4  # todo: try a bigger batch size -> they probably had VRAM constraints

    assert os.path.exists(dataset_path), f'MS COCO missing. Download the dataset using resource_downloader.py script.'
    os.makedirs(model_binaries_path, exist_ok=True)

    #
    # Modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    # todo: experiment with weights here
    parser.add_argument("--style_img_name", type=str, help="style image name that will be used for training", default='mosaic.jpg')
    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e0)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=3e5)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e-6)
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs ", default=1)
    parser.add_argument("--subset_size", type=int, help="number of MS COCO images to use, default is all (~83k)(specified by None)", default=10000)
    parser.add_argument("--log_freq", type=int, help="logging to output console frequency", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="how often to save the checkpoint model", default=None)
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

