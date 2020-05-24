import os
import argparse
import time

import torch
from torch.optim import Adam
import git

from models.definitions.perceptual_loss_net import PerceptualLossNet
from models.definitions.transformer_net import TransformerNet
import utils.utils as utils


def train(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = utils.get_training_data_loader(training_config)

    transformer_net = TransformerNet().train().to(device)
    perceptual_loss_net = PerceptualLossNet(requires_grad=False).to(device)

    optimizer = Adam(transformer_net.parameters())

    style_img_path = os.path.join(training_config['style_images_path'], training_config['style_img_name'])
    style_img = utils.prepare_img(style_img_path, target_shape=None, device=device, repeat=training_config['batch_size'])
    style_img_set_of_feature_maps = perceptual_loss_net(style_img)
    target_style_representation = [utils.gram_matrix(x) for x in style_img_set_of_feature_maps]

    acc_content_loss, acc_style_loss = [0., 0.]
    for epoch in range(training_config['num_of_epochs']):
        for batch_id, (content_batch, _) in enumerate(train_loader):
            # We want to train the transformer net to work on [0, 255] imagery
            content_batch = content_batch.to(device)
            stylized_batch = transformer_net(content_batch)

            # We need to normalize these because the perceptual loss net is VGG16 and it was trained on normalized imgs
            content_batch = utils.normalize_batch(content_batch)
            stylized_batch = utils.normalize_batch(stylized_batch)

            features_content_batch = perceptual_loss_net(content_batch)
            features_stylized_batch = perceptual_loss_net(stylized_batch)

            current_content_representation = features_stylized_batch.relu2_2
            target_content_representation = features_content_batch.relu2_2
            content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

            # todo: we have batch now debug this part
            style_loss = 0.0
            current_style_representation = [utils.gram_matrix(x) for x in features_stylized_batch]
            for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
                style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
            style_loss /= len(target_style_representation)

            # todo: add total variation
            total_loss = training_config['content_weight'] * content_loss + training_config['style_weight'] * style_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc_content_loss += content_loss.item()
            acc_style_loss += style_loss.item()

            #
            # Logging and checkpoint creation
            #
            if training_config['log_interval'] is not None and (batch_id + 1) % training_config['log_interval'] == 0:
                print(f'{time.ctime()} : Epoch={epoch + 1} : [{batch_id}/{len(train_loader)}] content: {acc_content_loss / (batch_id + 1)} style: {acc_style_loss / (batch_id + 1)} total: {(acc_content_loss + acc_style_loss) / (batch_id + 1)}')

            if training_config['checkpoint_freq'] is not None and (batch_id + 1) % training_config['checkpoint_freq'] == 0:
                ckpt_model_name = f"ckpt_style_{training_config['style_img_name'].split('.')[0]}_cw_{str(training_config['content_weight'])}_sw_{str(training_config['style_weight'])}_epoch_{epoch}_batch_{batch_id}.pth"
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

    model_name = f"style_{training_config['style_img_name'].split('.')[0]}_datapoints_{num_of_datapoints}_cw_{str(training_config['content_weight'])}_sw_{str(training_config['style_weight'])}.pth"
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
    parser.add_argument("--style_img_name", type=str, help="style image name that will be used for training", default='vg_starry_night.jpg')
    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=1e10)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs ", default=2)
    parser.add_argument("--subset_size", type=int, help="number of MS COCO images to use, default is all (~83k)(specified by None)", default=None)
    parser.add_argument("--log_freq", type=int, help="logging to output console frequency", default=None)
    parser.add_argument("--checkpoint_freq", type=int, help="how often to save the checkpoint model", default=500)
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

