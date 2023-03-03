import cv2
import os
import argparse


def create_video_from_images(images, fps, scale, path):

    for (i, image) in enumerate(images):

        img = cv2.imread(image)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        h, w, _ = img.shape
        if i == 0:
            # Create a VideoWriter object for mp4 video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(f"{path}/video.mp4", fourcc, fps, (w, h))

        out.write(img)

    # Release everything if job is finished
    out.release()


def main(fps, scale, path, interval):

    # Read all the images in the folder
    images = [img for img in os.listdir(path) if img.endswith(".png")]

    # Sort the images by name index
    images.sort(key=lambda x: int(x[:-4].split("_")[-1]))

    images = [os.path.join(path, img) for img in images]

    if interval:
        interval = interval.split(":")
        images = images[int(interval[0]) : int(interval[1])]

    create_video_from_images(images=images, fps=fps, scale=scale, path=path)


if __name__ == "__main__":

    arg = argparse.ArgumentParser()
    arg.add_argument("--path", type=str, default="images")
    arg.add_argument("--fps", type=int, default=20)
    arg.add_argument("--scale", type=float, default=1)
    arg.add_argument("--interval", type=str, default="")
    args = arg.parse_args()
    main(args.fps, args.scale, args.path, args.interval)
