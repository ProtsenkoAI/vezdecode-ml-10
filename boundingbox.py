import sys
import easyocr
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import shutil

import os
import numpy as np
from PIL import Image
from classifier import Classifier
from non_maximal_supression import non_max_suppression_fast


def plot(image, bounding_box):
	im = np.array(Image.open(image), dtype=np.uint8)
	fig,ax = plt.subplots(1)
	ax.imshow(im)

	# Create a Rectangle patch
	for x, y, height, width in bounding_box:
		# x, y = l[0]
		# height = abs(l[0][0] - l[1][0])
		# width = abs(l[0][1] - l[2][1])
		rect = patches.Rectangle((x,y),height,width,linewidth=1,edgecolor='r',facecolor='none')
		# Add the patch to the Axes
		ax.add_patch(rect)
	
	plt.axis('off')
	plt.savefig(image.parent / "bb" / (image.parts[-1]), dpi=300, bbox_inches='tight')


def remove_non_math(clf, boxes_data):
	return list(filter(lambda x: clf.ismath(x[1]), boxes_data))


def main():

	# classifier to detect math vs non-math text
	clf = Classifier()
	reader = easyocr.Reader(['en'])
	image_pth = sys.argv[1]
	out_dir_pth = sys.argv[2]
	shutil.rmtree(out_dir_pth, ignore_errors=True)
	os.makedirs(out_dir_pth	)

	print("Processing", image_pth)
	boxes_data = reader.readtext(str(image_pth),
			batch_size=16,
			beamWidth=1,
			link_threshold=0.1,
			low_text=0.4,
			height_ths=0.9,
			ycenter_ths=0.9
	)

	boxes_data = remove_non_math(clf, boxes_data)
	boxes = non_max_suppression_fast(
		np.array([(l[0][0], l[0][1], l[2][0], l[2][1]) for l, _, _ in boxes_data]),
		overlapThresh=0.1
	)
	# boxes = [(x1, y1, (x2-x1), (y2-y1)) for x1, y1, x2, y2 in boxes]

	image = Image.open(str(image_pth))

	marg = 10
	size_thresh = 20
	for idx, (x_min, y_min, x_max, y_max) in enumerate(boxes):
		if abs(x_min - x_max) >= size_thresh and abs(y_min - y_max) >= size_thresh:
			formula_crop = image.crop((x_min - marg, y_min - marg, x_max + marg, y_max + marg))
			formula_crop.save(os.path.join(out_dir_pth, f"{idx}.jpeg"))


	# plot(image, boxes)
	#
	# # printing
	# table = []
	# for _, text, confidence in boxes_data:
	# 	table.append([text, 1])
	# f = open(image.parent / "bb" / (image.parts[-1].replace("jpg", "txt")), "w")
	# print(tabulate(table, headers=["Text", "Label"]), file=f)


if __name__ == '__main__':
	main()

