#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Video compilation
'''

from moviepy.editor import ImageSequenceClip
import argparse
import os
import warning

IMAGE_EXT = ['jpeg', 'gif', 'png', 'jpg']


def main(args):

    # convert file folder into list firltered for image file types
    image_list = sorted([os.path.join(args.image_folder, image_file)
                        for image_file in os.listdir(args.image_folder)])

    image_list = [image_file for image_file in image_list if os.path.splitext(image_file)[1][1:].lower() in IMAGE_EXT]

    #two methods of naming output video to handle varying environemnts
    video_file_1 = args.image_folder + '.mp4'
    video_file_2 = args.image_folder + 'output_video.mp4'

    print(f"Creating video {args.image_folder}, FPS={args.fps}")
    clip = ImageSequenceClip(image_list, fps=args.fps)

    try:
        clip.write_videofile(video_file_1)
    except Exception as e:
        warning.warn(f'First video clip saving method failed: {e}')
        clip.write_videofile(video_file_2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driving session video creation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_folder", type=str, default='',
                        help="Path to image folder. The video will be created from these images.")
    parser.add_argument("--fps", type=int, default=60, help="FPS (Frames per second) setting for the video.")

    args = parser.parse_args()
    main(args)
