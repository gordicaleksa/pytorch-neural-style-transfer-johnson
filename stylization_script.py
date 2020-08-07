import os
import argparse


import torch
from torch.utils.data import DataLoader


import utils.utils as utils
from models.definitions.transformer_net import TransformerNet


def stylize_static_image(inference_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the model - load the weights and put the model into evaluation mode
    stylization_model = TransformerNet().to(device)
    training_state = torch.load(os.path.join(inference_config["model_binaries_path"], inference_config["model_name"]))
    state_dict = training_state["state_dict"]
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()

    if inference_config['verbose']:
        utils.print_model_metadata(training_state)

    with torch.no_grad():
        if os.path.isdir(inference_config['content_input']):  # do a batch stylization (every image in the directory)
            img_dataset = utils.SimpleDataset(inference_config['content_input'], inference_config['img_width'])
            img_loader = DataLoader(img_dataset, batch_size=inference_config['batch_size'])

            try:
                processed_imgs_cnt = 0
                for batch_id, img_batch in enumerate(img_loader):
                    processed_imgs_cnt += len(img_batch)
                    if inference_config['verbose']:
                        print(f'Processing batch {batch_id + 1} ({processed_imgs_cnt}/{len(img_dataset)} processed images).')

                    img_batch = img_batch.to(device)
                    stylized_imgs = stylization_model(img_batch).to('cpu').numpy()
                    for stylized_img in stylized_imgs:
                        utils.save_and_maybe_display_image(inference_config, stylized_img, should_display=False)
            except Exception as e:
                print(e)
                print(f'Consider making the batch_size (current = {inference_config["batch_size"]} images) or img_width (current = {inference_config["img_width"]} px) smaller')
                exit(1)

        else:  # do stylization for a single image
            content_img_path = os.path.join(inference_config['content_images_path'], inference_config['content_input'])
            content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
            stylized_img = stylization_model(content_image).to('cpu').numpy()[0]
            utils.save_and_maybe_display_image(inference_config, stylized_img, should_display=inference_config['should_not_display'])


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
    # Put image name or directory containing images (if you'd like to do a batch stylization on all those images)
    parser.add_argument("--content_input", type=str, help="Content image(s) to stylize", default='taj_mahal.jpg')
    parser.add_argument("--batch_size", type=int, help="Batch size used only if you set content_input to a directory", default=5)
    parser.add_argument("--img_width", type=int, help="Resize content image to this width", default=500)
    parser.add_argument("--model_name", type=str, help="Model binary to use for stylization", default='mosaic_4e5_e2.pth')

    # Less frequently used arguments
    parser.add_argument("--should_not_display", action='store_false', help="Should display the stylized result")
    parser.add_argument("--verbose", action='store_true', help="Print model metadata (how the model was trained) and where the resulting stylized image was saved")
    parser.add_argument("--redirected_output", type=str, help="Overwrite default output dir. Useful when this project is used as a submodule", default=None)
    args = parser.parse_args()

    # if redirected output is not set when doing batch stylization set to default image output location
    if os.path.isdir(args.content_input) and args.redirected_output is None:
        args.redirected_output = output_images_path

    # Wrapping inference configuration into a dictionary
    inference_config = dict()
    for arg in vars(args):
        inference_config[arg] = getattr(args, arg)
    inference_config['content_images_path'] = content_images_path
    inference_config['output_images_path'] = output_images_path
    inference_config['model_binaries_path'] = model_binaries_path

    stylize_static_image(inference_config)
