import os
import argparse

import torch

import utils.utils as utils
from models.definitions.transformer_net import TransformerNet


def stylize_static_image(inference_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img_path = os.path.join(inference_config['content_images_path'], inference_config['content_img_name'])
    content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)

    # load the weights and set the model to evaluation mode
    stylization_model = TransformerNet().to(device)
    training_state = torch.load(os.path.join(inference_config["model_binaries_path"], inference_config["model_name"]))
    state_dict = training_state["state_dict"]
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()

    if inference_config['verbose']:
        utils.print_model_metadata(training_state)

    with torch.no_grad():
        stylized_img = stylization_model(content_image).to('cpu').numpy()[0]
        utils.save_and_maybe_display_image(inference_config, stylized_img, should_display=inference_config['should_display'])


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    content_images_path = os.path.join(os.path.dirname(__file__), 'data', 'content-images')
    output_images_path = os.path.join(os.path.dirname(__file__), 'data', 'output-images')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')

    assert utils.dir_contains_only_models(model_binaries_path), f'Model directory should contain only model binaries.'
    os.makedirs(output_images_path, exist_ok=True)

    #
    # Modifiable args - feel free to play with these
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="Content image to stylize", default='taj_mahal.jpg')
    parser.add_argument("--img_width", type=int, help="Resize content image to this width", default=500)
    parser.add_argument("--model_name", type=str, help="Model binary to use for stylization", default='mosaic_4e5_e2.pth')

    # Less important arguments
    parser.add_argument("--should_display", action='store_false', help="Should display the stylized result (default true)")
    parser.add_argument("--verbose", action='store_false', help="Print model metadata (how the model was trained) and where the resulting stylized image was saved (default true)")
    parser.add_argument("--redirected_output", type=str, help="Overwrite default output dir. Useful when this project is used as a submodule", default=None)
    args = parser.parse_args()

    # Wrapping inference configuration into a dictionary
    inference_config = dict()
    for arg in vars(args):
        inference_config[arg] = getattr(args, arg)
    inference_config['content_images_path'] = content_images_path
    inference_config['output_images_path'] = output_images_path
    inference_config['model_binaries_path'] = model_binaries_path

    stylize_static_image(inference_config)
