import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Sampler
import torch


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_MEAN_1 = [0.485, 0.456, 0.406]
IMAGENET_STD_1 = [0.229, 0.224, 0.225]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def post_process(dump_img):
    mean = np.asarray(IMAGENET_MEAN_1).reshape(-1, 1, 1)
    std = np.asarray(IMAGENET_STD_1).reshape(-1, 1, 1)
    dump_img = (dump_img * std) + mean
    dump_img = (np.clip(dump_img, 0., 1.) * 255).astype(np.uint8)
    dump_img = np.moveaxis(dump_img, 0, 2)
    return dump_img


def save_and_maybe_display(inference_config, dump_img, should_display=False):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array got {type(dump_img)}.'

    dump_img = post_process(dump_img)
    dump_img_name = inference_config['content_img_name'].split('.')[0] + '_' + str(inference_config['img_height']) + '_' + inference_config['model_name'] + '.jpg'
    cv.imwrite(os.path.join(inference_config['output_images_path'], dump_img_name), dump_img[:, :, ::-1])  # ::-1 because opencv works with bgr...

    if should_display:
        plt.imshow(dump_img)
        plt.show()


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts bgr (opencv format...) into rgb

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the height
            ratio = target_shape / img.shape[0]
            width = int(img.shape[1] * ratio)
            img = cv.resize(img, (width, target_shape), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def prepare_img(img_path, target_shape, device, repeat=1, should_normalize=True, is_255_range=False):
    img = load_image(img_path, target_shape=target_shape)

    transform_list = [transforms.ToTensor()]
    if is_255_range:
        transform_list.append(transforms.Lambda(lambda x: x.mul(255)))
    if should_normalize:
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL) if is_255_range else transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))
    transform = transforms.Compose(transform_list)

    img = transform(img).to(device)
    img = img.repeat(repeat, 1, 1, 1)

    return img


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


def get_training_data_loader(training_config):
    transform = transforms.Compose([
        transforms.Resize(training_config['image_size']),
        transforms.CenterCrop(training_config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
        # transforms.Lambda(lambda x: x.mul(255)),
        # transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])
    train_dataset = datasets.ImageFolder(training_config['dataset_path'], transform)
    sampler = SequentialSubsetSampler(train_dataset, training_config['subset_size'])
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], sampler=sampler)
    print(f'Using {len(train_loader)*training_config["batch_size"]*training_config["num_of_epochs"]} datapoints (MS COCO images) for transformer network training.')
    return train_loader


# def special_preprocessing(img_batch):
#     img_batch += 1.0
#     print(torch.max(img_batch), torch.min(img_batch))
#     img_batch /= 2.0
#     print(torch.max(img_batch), torch.min(img_batch))
#     img_batch = transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)(img_batch)
#     print(torch.max(img_batch), torch.min(img_batch))
#     return img_batch


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def normalize_batch(batch, is_255_range=True):
    # todo: add some assert here
    # Normalize using ImageNet's mean
    if is_255_range:
        batch /= 255.0
        mean = batch.new_tensor(IMAGENET_MEAN_1).view(-1, 1, 1)
        std = batch.new_tensor(IMAGENET_STD_1).view(-1, 1, 1)
        return (batch - mean) / std
    else:
        mean = batch.new_tensor(IMAGENET_MEAN_255).view(-1, 1, 1)
        return batch - mean


def total_variation(img_batch):
    batch_size = img_batch.shape[0]
    return (torch.sum(torch.abs(img_batch[:, :, :, :-1] - img_batch[:, :, :, 1:])) +
            torch.sum(torch.abs(img_batch[:, :, :-1, :] - img_batch[:, :, 1:, :]))) / batch_size


def print_header(training_config):
    print(f'Training network to learn the style of {training_config["style_img_name"]} style image.')
    print('*' * 70)
    print(f'Hyperparams: content_weight={training_config["content_weight"]}, style_weight={training_config["style_weight"]} and tv_weight={training_config["tv_weight"]}')
    print('*' * 70)
    print(f'Logging frequency every {training_config["log_freq"]} batches.')


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
