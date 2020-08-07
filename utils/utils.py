import os
import re


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import git


IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_255 = np.array([123.675, 116.28, 103.53])
# Usually when normalizing 0..255 images only mean-normalization is performed -> that's why standard dev is all 1s here
IMAGENET_STD_NEUTRAL = np.array([1, 1, 1])


class SimpleDataset(Dataset):
    def __init__(self, img_dir, target_width):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]

        h, w = load_image(self.img_paths[0]).shape[:2]
        img_height = int(h * (target_width / w))
        self.target_width = target_width
        self.target_height = img_height

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_image(self.img_paths[idx], target_shape=(self.target_height, self.target_width))
        tensor = self.transform(img)
        return tensor


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def prepare_img(img_path, target_shape, device, batch_size=1, should_normalize=True, is_255_range=False):
    img = load_image(img_path, target_shape=target_shape)

    transform_list = [transforms.ToTensor()]
    if is_255_range:
        transform_list.append(transforms.Lambda(lambda x: x.mul(255)))
    if should_normalize:
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL) if is_255_range else transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))
    transform = transforms.Compose(transform_list)

    img = transform(img).to(device)
    img = img.repeat(batch_size, 1, 1, 1)

    return img


def post_process_image(dump_img):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy image got {type(dump_img)}'

    mean = IMAGENET_MEAN_1.reshape(-1, 1, 1)
    std = IMAGENET_STD_1.reshape(-1, 1, 1)
    dump_img = (dump_img * std) + mean  # de-normalize
    dump_img = (np.clip(dump_img, 0., 1.) * 255).astype(np.uint8)
    dump_img = np.moveaxis(dump_img, 0, 2)
    return dump_img


def get_next_available_name(input_dir):
    img_name_pattern = re.compile(r'[0-9]{6}\.jpg')
    candidates = [candidate for candidate in os.listdir(input_dir) if re.fullmatch(img_name_pattern, candidate)]

    if len(candidates) == 0:
        return '000000.jpg'
    else:
        latest_file = sorted(candidates)[-1]
        prefix_int = int(latest_file.split('.')[0])
        return f'{str(prefix_int + 1).zfill(6)}.jpg'


def save_and_maybe_display_image(inference_config, dump_img, should_display=False):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array got {type(dump_img)}.'

    dump_img = post_process_image(dump_img)
    if inference_config['img_width'] is None:
        inference_config['img_width'] = dump_img.shape[0]

    if inference_config['redirected_output'] is None:
        dump_dir = inference_config['output_images_path']
        dump_img_name = os.path.basename(inference_config['content_input']).split('.')[0] + '_width_' + str(inference_config['img_width']) + '_model_' + inference_config['model_name'].split('.')[0] + '.jpg'
    else:  # useful when this repo is used as a utility submodule in some other repo like pytorch-naive-video-nst
        dump_dir = inference_config['redirected_output']
        os.makedirs(dump_dir, exist_ok=True)
        dump_img_name = get_next_available_name(inference_config['redirected_output'])

    cv.imwrite(os.path.join(dump_dir, dump_img_name), dump_img[:, :, ::-1])  # ::-1 because opencv expects BGR (and not RGB) format...

    # Don't print this information in batch stylization mode
    if inference_config['verbose'] and not os.path.isdir(inference_config['content_input']):
        print(f'Saved image to {dump_dir}.')

    if should_display:
        plt.imshow(dump_img)
        plt.show()


class SequentialSubsetSampler(Sampler):
    r"""Samples elements sequentially, always in the same order from a subset defined by size.

    Arguments:
        data_source (Dataset): dataset to sample from
        subset_size: defines the subset from which to sample from
    """

    def __init__(self, data_source, subset_size):
        assert isinstance(data_source, Dataset) or isinstance(data_source, datasets.ImageFolder)
        self.data_source = data_source

        if subset_size is None:  # if None -> use the whole dataset
            subset_size = len(data_source)
        assert 0 < subset_size <= len(data_source), f'Subset size should be between (0, {len(data_source)}].'
        self.subset_size = subset_size

    def __iter__(self):
        return iter(range(self.subset_size))

    def __len__(self):
        return self.subset_size


def get_training_data_loader(training_config, should_normalize=True, is_255_range=False):
    """
        There are multiple ways to make this feed-forward NST working,
        including using 0..255 range (without any normalization) images during transformer net training,
        keeping the options if somebody wants to play and get better results.
    """
    transform_list = [transforms.Resize(training_config['image_size']),
                      transforms.CenterCrop(training_config['image_size']),
                      transforms.ToTensor()]
    if is_255_range:
        transform_list.append(transforms.Lambda(lambda x: x.mul(255)))
    if should_normalize:
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL) if is_255_range else transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.ImageFolder(training_config['dataset_path'], transform)
    sampler = SequentialSubsetSampler(train_dataset, training_config['subset_size'])
    training_config['subset_size'] = len(sampler)  # update in case it was None
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], sampler=sampler, drop_last=True)
    print(f'Using {len(train_loader)*training_config["batch_size"]*training_config["num_of_epochs"]} datapoints ({len(train_loader)*training_config["num_of_epochs"]} batches) (MS COCO images) for transformer network training.')
    return train_loader


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


# Not used atm, you'd want to use this if you choose to go with 0..255 images in the training loader
def normalize_batch(batch):
    batch /= 255.0
    mean = batch.new_tensor(IMAGENET_MEAN_1).view(-1, 1, 1)
    std = batch.new_tensor(IMAGENET_STD_1).view(-1, 1, 1)
    return (batch - mean) / std


def total_variation(img_batch):
    batch_size = img_batch.shape[0]
    return (torch.sum(torch.abs(img_batch[:, :, :, :-1] - img_batch[:, :, :, 1:])) +
            torch.sum(torch.abs(img_batch[:, :, :-1, :] - img_batch[:, :, 1:, :]))) / batch_size


def print_header(training_config):
    print(f'Learning the style of {training_config["style_img_name"]} style image.')
    print('*' * 80)
    print(f'Hyperparams: content_weight={training_config["content_weight"]}, style_weight={training_config["style_weight"]} and tv_weight={training_config["tv_weight"]}')
    print('*' * 80)

    if training_config["console_log_freq"]:
        print(f'Logging to console every {training_config["console_log_freq"]} batches.')
    else:
        print(f'Console logging disabled. Change console_log_freq if you want to use it.')

    if training_config["checkpoint_freq"]:
        print(f'Saving checkpoint models every {training_config["checkpoint_freq"]} batches.')
    else:
        print(f'Checkpoint models saving disabled.')

    if training_config['enable_tensorboard']:
        print('Tensorboard enabled.')
        print('Run "tensorboard --logdir=runs --samples_per_plugin images=50" from your conda env')
        print('Open http://localhost:6006/ in your browser and you\'re ready to use tensorboard!')
    else:
        print('Tensorboard disabled.')
    print('*' * 80)


def get_training_metadata(training_config):
    num_of_datapoints = training_config['subset_size'] * training_config['num_of_epochs']
    training_metadata = {
        "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "content_weight": training_config['content_weight'],
        "style_weight": training_config['style_weight'],
        "tv_weight": training_config['tv_weight'],
        "num_of_datapoints": num_of_datapoints
    }
    return training_metadata


def print_model_metadata(training_state):
    print('Model training metadata:')
    for key, value in training_state.items():
        if key != 'state_dict' and key != 'optimizer_state':
            print(key, ':', value)


def dir_contains_only_models(path):
    assert os.path.exists(path), f'Provided path: {path} does not exist.'
    assert os.path.isdir(path), f'Provided path: {path} is not a directory.'
    list_of_files = os.listdir(path)
    assert len(list_of_files) > 0, f'No models found, use training_script.py to train a model or download pretrained models via resource_downloader.py.'
    for f in list_of_files:
        if not (f.endswith('.pt') or f.endswith('.pth')):
            return False

    return True


# Count how many trainable weights the model has <- just for having a feeling for how big the model is
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
