import argparse
import os, shutil
import joblib
import imageio
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import trange
from rllab import config

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--name', type=str, default='video', help='Name of output file without mp4')
    parser.add_argument('--max_path_length', type=int, default=500, help='Max length of rollout')
    parser.add_argument('--fps', type=float, default=24, help='frames per second')
    parser.add_argument('--period', type=int, default=-1)
    parser.add_argument('--random_latent', '-random_lat', action='store_true')
    parser.add_argument('--fixed_latent', '-fixed_lat', type=int, default=-1)
    args = parser.parse_args()

    if os.path.isdir("temp_images"):
        shutil.rmtree("temp_images")
    os.makedirs("temp_images")
    filenames = []
    if args.random_latent:
        assert args.period != -1

    with tf.Session() as sess:
        data = joblib.load(os.path.join(config.PROJECT_PATH, args.file))
        policy = data['policy']
        if args.period != -1:
            policy.period = args.period
        env = data['env']

        obs = env.reset()
        image_array = env.render(mode="rgb_array")
        height, width, _ = image_array.shape
        if args.fixed_latent != -1:
            curr_latent = policy.outer_action_space.flatten(args.fixed_latent)
            policy.low_policy.set_pre_fix_latent(curr_latent)
            policy.low_policy.reset()
        for i in trange(args.max_path_length):
            # if i % 100 == 0:
            #     print(i)
            if args.random_latent:
                if i % args.period == 0:
                    curr_latent = policy.outer_action_space.flatten(np.random.randint(0, 6))
                    policy.low_policy.set_pre_fix_latent(curr_latent)
                    policy.low_policy.reset()
                action, action_infos = policy.low_policy.get_action(obs)
            elif args.fixed_latent != -1:
                action, action_infos = policy.low_policy.get_action(obs)
            else:
                action, action_infos = policy.get_action(obs)
            # create Image object with the input image
            im = Image.fromarray(image_array)
            draw = ImageDraw.Draw(im)
            font = ImageFont.truetype('Roboto-Bold.ttf', size=40)
            # starting position of the message
            (x, y) = (width - 175, height - 85)
            name = 'Latent: {}'.format(np.argmax(action_infos['latents']))
            color = 'rgb(65, 101, 244)'
            draw.text((x, y), name, fill=color, font=font)
            filename = "temp_images/{}.png".format(i)
            im.save(filename)
            filenames.append(filename)
            obs, reward, done, info = env.step(action)
            image_array = env.render(mode="rgb_array")

        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        if args.fixed_latent != -1:
            video_name = 'videos/latent{}.mp4'.format(args.fixed_latent)
        else:
            video_name = 'videos/{}.mp4'.format(args.name)
        imageio.mimsave(os.path.join(config.PROJECT_PATH, video_name), images, fps=args.fps)
        shutil.rmtree("temp_images")
