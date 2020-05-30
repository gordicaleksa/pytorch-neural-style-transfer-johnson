## Fast Neural Style Transfer (feed-forward method) :zap::computer: + :art: = :heart:
This repo contains a concise PyTorch implementation of the original feed-forward NST paper (:link: [Johnson et al.](https://arxiv.org/pdf/1603.08155.pdf)).

Checkout my implementation of the original NST (optimization method) paper ([Gatys et al.](https://github.com/gordicaleksa/pytorch-neural-style-transfer)).

It's an accompanying repo for [this video series on YouTube](https://www.youtube.com/watch?v=S78LQebx6jo&list=PLBoQnSflObcmbfshq9oNs41vODgXG-608)

<p align="left">
<a href="https://www.youtube.com/watch?v=S78LQebx6jo" target="_blank"><img src="https://img.youtube.com/vi/S78LQebx6jo/0.jpg" 
alt="NST Intro" width="480" height="360" border="10" /></a>
</p>

### Why yet another Fast NST (feed-forward method) repo?
It's the **cleanest and most concise** NST repo that I know of + it's written in **PyTorch!** :heart:

My idea :bulb: is to make the code so simple and well commented that you can use it as a **first step on your NST learning journey** before any other blog, course, book or research paper. :books:

I've included automatic MS COCO dataset and pretrained models download script - so you can either **instantaneously run it** and get the results (:art: stylized images) using pretrained models **or start training/experimenting with your own models**. :rocket:

## Examples

## Setup

1. Run `conda env create` from project directory.
2. Run `activate pytorch-nst-fast`

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies.

-----

PyTorch package will pull some version of CUDA with it, but it is highly recommended that you install system-wide CUDA beforehand, mostly because of GPU drivers. I also recommend using Miniconda installer as a way to get conda on your system. 

Follow through points 1 and 2 of [this setup](https://github.com/Petlja/PSIML/blob/master/docs/MachineSetup.md) and use the most up-to-date versions of Miniconda and CUDA/cuDNN.
(I recommend CUDA 10.1 or 10.2 as those are compatible with PyTorch 1.5, which is used in this repo, and newest compatible cuDNN)

## Usage

### Stylization

1. Download pretrained models, run: `python resource_downloader.py` (it'll download pretrained models by default)
2. Run `python stylization_script.py` (it's got default content image and model set)

That's it! If you want more flexibility (and I guess you do) there's a couple more nuggets of info:

Full command is: `python stylization_script.py --content_img_name <name> --img_height <height in px> --model_name <name>`

You just need to specify **names**, it pulls content images and models from default directories:
1. content images default dir: `/data/content-images/`
2. model binaries default dir: `/models/binaries/`

So all **you** got to do is place images and models there and you can use them.

After you run resource_downloader.py script binaries dir will be pre-populated with 4 pretrained models.

todo: add some image to this section (watch for size, .jpg)

### Training your own models

1. Download MS COCO dataset, run `python resource_downloader.py -r mscoco_dataset`
2. Run `python training_script.py --style_img_name <name>`

Now that will probably actually work!

It will periodically dump checkpoint models to `models/checkpoints/` and the final model to `models/binaries/` by default.

I strongly recommend playing with these 2 params:
1. style_weight - I always kept it in [1e5, 9e5], you may have to tweek it for your specific style image a little bit
2. subset_size - Usually 32000 images do the job (that's 8k batches) - you'll need to monitor tensorboard to figure out if your curves are saturating at that point or not

That bring us to the next section!

### Visualizations



## Debugging

## Experiment

## Acknowledgements

I found these repos useful: (while developing this one)
* [fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style) (PyTorch, feed-forward method)
* [pytorch-neural-style-transfer](https://github.com/gordicaleksa/pytorch-neural-style-transfer) (PyTorch, optimization method)
* [original J.Johnson's repo](https://github.com/jcjohnson/fast-neural-style) (Torch, written in Lua)

I found some of the content/style images I was using here:
* [style/artistic images](https://www.rawpixel.com/board/537381/vincent-van-gogh-free-original-public-domain-paintings?sort=curated&mode=shop&page=1)
* [awesome figures pic](https://www.pexels.com/photo/action-android-device-electronics-595804/)

Other images are now already classics in the NST world.

## Citation

If you find this code useful for your research, please cite the following:

```
@misc{Gordić2020nst-fast,
  author = {Gordić, Aleksa},
  title = {pytorch-nst-feedforward},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-nst-feedforward}},
}
```