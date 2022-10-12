import cv2
import os
import argparse

def main(path):
	# Create a VideoWriter object for mp4 video
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(f"{path}/video.mp4", fourcc, 20.0, (1200, 600))

	# Read all the images in the folder
	images = [img for img in os.listdir(path) if img.endswith(".png")]

	# Sort the images by name index
	images.sort(key=lambda x: int(x[:-4]))

	for image in images:
		img = cv2.imread(os.path.join(path, image))
		img = cv2.resize(img, (1200, 600))
		out.write(img)
	
	# Release everything if job is finished
	out.release()

if __name__ == "__main__":

	arg = argparse.ArgumentParser()
	arg.add_argument("--path", type=str, default="images")
	args = arg.parse_args()
	main(args.path)


