import os
import time
import re
import argparse

import torch
import numpy as np

import utils.utils as utils
from models.definitions.transformer_net import TransformerNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stylize_static_image(inference_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img_path = os.path.join(inference_config['content_images_path'], inference_config['content_img_name'])
    content_image = utils.prepare_img(content_img_path, inference_config['img_height'], device, should_normalize=False)

    stylization_model = TransformerNet().to(device)
    state_dict = torch.load(os.path.join(inference_config["model_binaries_path"], inference_config["model_name"]))[
        "state_dict"]
    # todo: debug if this is actually a thing
    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            print('wow. so much wow.')
            del state_dict[k]
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()

    with torch.no_grad():
        print(count_parameters(stylization_model))
        ts = time.time()
        # todo: debug why this one is failing
        stylized_img = stylization_model(content_image).to('cpu').numpy()[0]
        print(np.max(stylized_img), np.min(stylized_img), stylized_img.shape)
        print(f"Infer finished in: {time.time()-ts:.3f}")
        # utils.save_image(training_config.output_image, output[0])


# def aac_filter(vid_name):
#     return vid_name.endswith('.aac')
#
#
# def mp4_filter(name):
#     return name.endswith('.mp4')
#
#
# def is_big_resolution(content_pics):
#     img_path = os.path.join(content_pics, os.listdir(content_pics)[0])
#     img = utils.load_image(img_path)
#     h, w = img.shape[:2]
#     return w == 1920
#
#
# def modify_paths(paths):
#     new_paths = []
#     for path in paths:
#         base, name = os.path.split(path)
#         name = '_res_' + name
#         new_path = os.path.join(base, name)
#         new_paths.append(new_path)
#
#     return new_paths
#
#
# def stylize_video(args):
#     video_dir_path = args.content_image
#     format = args.image_format
#     print('Processing {}.'.format(video_dir_path))
#     audio_file_name = list(filter(aac_filter, os.listdir(video_dir_path)))[0]
#     in_audio_path = os.path.join(video_dir_path, audio_file_name)
#     video_name = list(filter(mp4_filter, os.listdir(video_dir_path)))[0].split('.')[0]
#     frames_path = os.path.join(video_dir_path, 'frames')
#     model_name = os.path.split(args.model)[1].split('.')[0]
#     dump_path = os.path.join(os.path.split(video_dir_path)[0], os.path.split(video_dir_path)[1] + '_' + model_name)
#
#     content_pics = os.path.join(frames_path, 'frames')
#     mask_dest = os.path.join(video_dir_path, 'processed_masks_refined')
#
#     device = torch.device("cuda" if args.cuda else "cpu")
#
#     content_transform = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Lambda(lambda x: x.mul(255))
#     ])
#     dataset = datasets.ImageFolder(frames_path, transform=content_transform)
#     num_of_frames = len(dataset)
#     using_big_res = is_big_resolution(content_pics)
#     batch_size = 3 if using_big_res else 14
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     print('Using batch size = {}'.format(batch_size))
#
#     dump_dest = os.path.join(dump_path, 'stylized')
#     second_style_path = args.second_style
#     final_dest_black = os.path.join(dump_path, 'combined_black')
#     final_dest_bkg = os.path.join(dump_path, 'combined_background')
#     final_dest_bkg_inv = os.path.join(dump_path, 'combined_background_inv')
#     os.makedirs(dump_dest, exist_ok=True)
#     os.makedirs(final_dest_black, exist_ok=True)
#     os.makedirs(final_dest_bkg, exist_ok=True)
#     os.makedirs(final_dest_bkg_inv, exist_ok=True)
#
#     if args.model.endswith(".onnx"):
#         # output = stylize_onnx_caffe2(content_image, args)
#         print('nicee.')
#     else:
#         with torch.no_grad():
#             style_model = TransformerNet()
#             state_dict = torch.load(args.model)
#             # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
#             for k in list(state_dict.keys()):
#                 if re.search(r'in\d+\.running_(mean|var)$', k):
#                     del state_dict[k]
#             style_model.load_state_dict(state_dict)
#             style_model.to(device)
#             if args.export_onnx:
#                 assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
#                 # output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
#             else:
#                 if len(os.listdir(dump_dest)) == 0:
#                     for i, (imgs, labels) in enumerate(loader):
#                         imgs = imgs.to(device)
#                         out_cpu_batch = style_model(imgs).cpu().numpy()
#                         ts = time.time()
#                         for j, styled_img in enumerate(out_cpu_batch):
#                             out_path = os.path.join(dump_dest, str(i*batch_size+j).zfill(4) + format)
#                             utils.save_image_from_vid(out_path, styled_img)
#                         print('{:04d}/{:04d} , processing batch took {:.3f}'.format((i+1)*batch_size, num_of_frames, time.time()-ts))
#                 else:
#                     print('Skipping, already stilyzed.')
#
#     ffmpeg_path = r'C:\tmp_data_dir\ffmpeg-4.2.2-win64-static\bin\ffmpeg.exe'
#     pic_path_pattern_blk = os.path.join(final_dest_black, 'combined_%04d' + format)
#     pic_path_pattern_bkg = os.path.join(final_dest_bkg, 'combined_%04d' + format)
#     pic_path_pattern_bkg_inv = os.path.join(final_dest_bkg_inv, 'combined_%04d' + format)
#     pic_path_pattern_stylized = os.path.join(dump_dest, '%04d' + format)
#     out_video_path_blk = os.path.join(final_dest_black, video_name + '_black_bkg_' + model_name + '.mp4')
#     out_video_path_bkg = os.path.join(final_dest_bkg, video_name + '_stylized_human_' + model_name + '.mp4')
#     out_video_path_bkg_inv = os.path.join(final_dest_bkg_inv, video_name + '_stylized_bkg_' + model_name + '.mp4')
#     out_video_path_stylized = os.path.join(dump_dest, video_name + '_full-style_' + model_name + '.mp4')
#
#     # if len(os.listdir(final_dest_black)) == 0:
#     #     for cnt, (path1, path2) in enumerate(zip(os.listdir(dump_dest), os.listdir(mask_dest))):
#     #         s_img_path = os.path.join(dump_dest, path1)
#     #         m_img_path = os.path.join(mask_dest, path2)
#     #         s_img = utils.load_image(s_img_path)
#     #         m_img = utils.load_image(m_img_path)
#     #         s_h, s_w = s_img.shape[:2]
#     #         m_img = cv.resize(m_img, (s_w, s_h))
#     #         mask = m_img == 0
#     #         # combined = np.einsum('ijk,ij->ijk', s_img, m_img)
#     #         s_img[mask] = 0
#     #         out_path = os.path.join(final_dest_black, 'combined_' + str(cnt).zfill(4) + format)
#     #         # print(out_path)
#     #         img = Image.fromarray(s_img)
#     #         img.save(out_path)
#     #
#     #         # plt.imshow(s_img)
#     #         # plt.show()
#     #     print('Done making black pics.')
#     # else:
#     #     print('Skipping, blacks already combined.')
#     #
#     # subprocess.call([ffmpeg_path, '-r', str(30), '-f', 'image2' , '-i', pic_path_pattern_blk, '-i', in_audio_path, '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', '-c:a', 'copy', out_video_path_blk])
#     # # free up some memory and just keep the videos
#     # if args.should_delete_images:
#     #   [os.remove(os.path.join(final_dest_black, file)) for file in os.listdir(final_dest_black) if file.endswith(format)]
#
#     # todo: this could be done on the GPU
#     ts = time.time()
#     if os.path.exists(second_style_path):
#         second_dir = second_style_path
#     else:
#         second_dir = content_pics
#     if len(os.listdir(final_dest_bkg)) == 0 and len(os.listdir(final_dest_bkg_inv)) == 0:
#         for cnt, (path1, path2, path3) in enumerate(zip(os.listdir(dump_dest), os.listdir(mask_dest), os.listdir(second_dir))):
#             s_img_path = os.path.join(dump_dest, path1)
#             m_img_path = os.path.join(mask_dest, path2)
#             c_img_path = os.path.join(second_dir, path3)
#             s_img = utils.load_image(s_img_path)
#             m_img = utils.load_image(m_img_path)
#             s_h, s_w = s_img.shape[:2]
#             m_img = cv.resize(m_img, (s_w, s_h))
#             c_img = utils.load_image(c_img_path)
#             s_img_inv = s_img.copy()
#
#             mask = m_img == 0
#             mask_inv = m_img == 255
#
#             s_img_inv[mask_inv] = c_img[mask_inv]
#             s_img[mask] = c_img[mask]
#
#             out_path = os.path.join(final_dest_bkg, 'combined_' + str(cnt).zfill(4) + format)
#             out_path_inv = os.path.join(final_dest_bkg_inv, 'combined_' + str(cnt).zfill(4) + format)
#             # print(out_path)
#             img = Image.fromarray(s_img)
#             img.save(out_path)
#
#             img_inv = Image.fromarray(s_img_inv)
#             img_inv.save(out_path_inv)
#
#             # plt.imshow(s_img)
#             # plt.show()
#         print('Done making background pics.')
#     else:
#         print('Skipping, background pics already combined.')
#     print('Combining with masks took {:.3f}'.format(time.time() - ts))
#
#     # after image2 it went like this: '-s', '1920x1080' '960x540'
#     subprocess.call([ffmpeg_path, '-r', str(30), '-f', 'image2', '-i', pic_path_pattern_bkg, '-i', in_audio_path, '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', '-c:a', 'copy', out_video_path_bkg])
#     subprocess.call([ffmpeg_path, '-r', str(30), '-f', 'image2', '-i', pic_path_pattern_bkg_inv, '-i', in_audio_path, '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', '-c:a', 'copy', out_video_path_bkg_inv])
#     subprocess.call([ffmpeg_path, '-r', str(30), '-f', 'image2', '-i', pic_path_pattern_stylized, '-i', in_audio_path, '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', '-c:a', 'copy', out_video_path_stylized])
#     print('Creating videos done.')
#
#     if not using_big_res:
#         resized_videos_paths = modify_paths([out_video_path_bkg, out_video_path_bkg_inv, out_video_path_stylized])
#         subprocess.call([ffmpeg_path, '-i', out_video_path_bkg, '-vf', 'scale=1920:1080', resized_videos_paths[0]])
#         subprocess.call([ffmpeg_path, '-i', out_video_path_bkg_inv, '-vf', 'scale=1920:1080', resized_videos_paths[1]])
#         subprocess.call([ffmpeg_path, '-i', out_video_path_stylized, '-vf', 'scale=1920:1080', resized_videos_paths[2]])
#         print('Done resizing videos')
#
#     if args.should_delete_images:
#         [os.remove(os.path.join(final_dest_bkg, file)) for file in os.listdir(final_dest_bkg) if file.endswith(format)]
#         [os.remove(os.path.join(final_dest_bkg_inv, file)) for file in os.listdir(final_dest_bkg_inv) if file.endswith(format)]
#         # todo: tmp disabled
#         # [os.remove(os.path.join(dump_dest, file)) for file in os.listdir(dump_dest) if file.endswith(format)]
#         print('Deleting images done.')
#     else:
#         print('Won"t delete images')


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    content_images_path = os.path.join(os.path.dirname(__file__), 'data', 'content-images')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    assert utils.dir_contains_only_models(model_binaries_path), f'Model directory should contain only model binaries.'
    model_names = os.listdir(model_binaries_path)  # list of available model binaries

    #
    # Modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image to stylize", default='figures.jpg')
    parser.add_argument("--img_height", type=int, help="resize content image to this height", default=None)
    parser.add_argument("--model_name", type=str, help="model binary to use for stylization", default=model_names[0])
    args = parser.parse_args()

    # Wrapping inference configuration into a dictionary
    inference_config = dict()
    for arg in vars(args):
        inference_config[arg] = getattr(args, arg)
    inference_config['model_binaries_path'] = model_binaries_path
    inference_config['content_images_path'] = content_images_path

    stylize_static_image(inference_config)
