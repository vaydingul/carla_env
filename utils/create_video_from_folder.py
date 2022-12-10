import cv2
import os
import argparse


def main(fps, path):
    

    # Read all the images in the folder
    images = [img for img in os.listdir(path) if img.endswith(".png")]

    # Sort the images by name index
    images.sort(key=lambda x: int(x[:-4]))

    for (i, image) in enumerate(images):
        
        img = cv2.imread(os.path.join(path, image))
        # img = cv2.resize(img, (1200, 600))
        h, w, _ = img.shape
        if i == 0:
            # Create a VideoWriter object for mp4 video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f"{path}/video.mp4", fourcc, fps, (w, h))
        
        out.write(img)

    # Release everything if job is finished
    out.release()


if __name__ == "__main__":

    arg = argparse.ArgumentParser()
    arg.add_argument("--path", type=str, default="images")
    arg.add_argument("--fps", type=int, default=20)
    args = arg.parse_args()
    main(args.fps, args.path)
