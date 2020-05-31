import zipfile
from torch.hub import download_url_to_file
import argparse
import os

# If the link is broken you can download the MS COCO 2014 dataset manually from http://cocodataset.org/#download
MS_COCO_2014_TRAIN_DATASET_PATH = r'http://images.cocodataset.org/zips/train2014.zip'  # ~13 GB after unzipping

PRETRAINED_MODELS_PATH = r'https://www.dropbox.com/s/fb39gscd1b42px1/pretrained_models.zip?dl=1'

DOWNLOAD_DICT = {
    'pretrained_models': PRETRAINED_MODELS_PATH,
    'mscoco_dataset': MS_COCO_2014_TRAIN_DATASET_PATH,
}
download_choices = list(DOWNLOAD_DICT.keys())


if __name__ == '__main__':
    #
    # Choose whether you want to download pretrained models or MSCOCO 2014 dataset
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", "-r", type=str, choices=download_choices,
                        help="specify whether you want to download ms coco dataset or pretrained models",
                        default=download_choices[0])
    args = parser.parse_args()

    # step1: download the resource to local filesystem
    remote_resource_path = DOWNLOAD_DICT[args.resource]
    print(f'Downloading from {remote_resource_path}')
    resource_tmp_path = args.resource + '.zip'
    download_url_to_file(remote_resource_path, resource_tmp_path)

    # step2: unzip the resource
    print(f'Started unzipping...')
    with zipfile.ZipFile(resource_tmp_path) as zf:
        local_resource_path = os.path.join(os.path.dirname(__file__), os.pardir)
        if args.resource == 'pretrained_models':
            local_resource_path = os.path.join(local_resource_path, 'models', 'binaries')
        else:
            local_resource_path = os.path.join(local_resource_path, 'data', 'mscoco')
        os.makedirs(local_resource_path, exist_ok=True)
        zf.extractall(path=local_resource_path)
    print(f'Unzipping to: {local_resource_path} finished.')

    # step3: remove the temporary resource file
    os.remove(resource_tmp_path)
    print(f'Removing tmp file {resource_tmp_path}.')
