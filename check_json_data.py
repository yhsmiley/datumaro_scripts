from pprint import pprint

from datumaro.components.project import Project
from datum_utils import num_img, num_img_with_annots, num_annots, print_extractor_info, compute_image_statistics, compute_ann_statistics


def check_data(json_path):
	# create Datumaro project
	project = Project()

	# add source
	project.add_source('src1', {'url': str(json_path), 'format': 'coco_instances'})

	# create a dataset
	dataset = project.make_dataset()
	print(f'{json_path.stem}')

	print(f'num images: {num_img(dataset)}')
	print(f'num images with annotations: {num_img_with_annots(dataset)}')
	print(f'num annotations: {num_annots(dataset)}')
	# print_extractor_info(dataset, indent=" ")
	# stats = {}
	# stats.update(compute_image_statistics(dataset))
	# stats.update(compute_ann_statistics(dataset))
	# pprint(stats)


if __name__ == '__main__':
	import argparse
	from pathlib import Path

	ap = argparse.ArgumentParser()
	group = ap.add_mutually_exclusive_group(required=True)
	group.add_argument('--json_path', help='annotations json path')
	group.add_argument('--annots_folder', help='folder path containing annotation jsons')
	args = ap.parse_args()

	if args.annots_folder:
		# doesnt recursively search in subfolders
		for i, json_path in enumerate(Path(args.annots_folder).iterdir()):
			if json_path.suffix == '.json':
				check_data(json_path)
	elif args.json_path:
		check_data(Path(args.json_path))
