import os

import cv2 as cv
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Sampler


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1].astype(np.float32)  # [:, :, ::-1] converts rgb into bgr (opencv contraint...)
    img /= 255.0  # get to [0, 1] range
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the height
            ratio = target_shape / img.shape[0]
            width = int(img.shape[1] * ratio)
            img = cv.resize(img, (width, target_shape), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)
    return img


def prepare_img(img_path, target_shape, device, repeat=1):
    img = load_image(img_path, target_shape=target_shape)

    # normalize using ImageNet's mean and std (VGG was trained on images normalized this way)
    # [0, 255] range works much better than [0, 1] range (VGG was again trained that way)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = transform(img).to(device)

    # todo: debug if this has the right shape and for repeat=1 is it == to unsqueeze 0
    img.repeat(repeat, 1, 1, 1)

    return img


class SequentialSubsetSampler(Sampler):
    r"""Samples elements sequentially, always in the same order from a subset defined by size.

    Arguments:
        data_source (Dataset): dataset to sample from
        subset_size: defines the subset from which to sample from
    """

    def __init__(self, data_source, subset_size):
        super().__init__()

        self.data_source = data_source
        if subset_size is None:  # if None -> use the whole dataset
            subset_size = len(data_source)
        assert 0 < subset_size <= len(data_source), f'Subset size should be between (0, {len(data_source)}].'
        self.subset_size = subset_size

    def __iter__(self):
        return iter(range(self.subset_size))

    def __len__(self):
        return len(self.subset_size)


def get_training_data_loader(training_config):
    transform = transforms.Compose([
        transforms.Resize(training_config['image_size']),
        transforms.CenterCrop(training_config['image_size']),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(training_config['dataset_path'], transform)
    sampler = SequentialSubsetSampler(train_dataset, training_config['subset_size'])
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], sampler=sampler)

    return train_loader


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def normalize_batch(batch):
    # Normalize using ImageNet's mean
    # todo: debug if this has correct shape
    mean = batch.new_tensor(IMAGENET_MEAN_255).view(-1, 1, 1)
    return batch - mean