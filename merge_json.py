import argparse
from pathlib import Path
from pprint import pprint

from datumaro.components.project import Project
from datum_utils import num_img, num_img_with_annots, num_annots, export_json, check_json_path


### WARNING: IMAGE FILE_NAME MUST BE UNIQUE!!! ###

ap = argparse.ArgumentParser()
group = ap.add_mutually_exclusive_group(required=True)
group.add_argument('--json_paths', nargs="+", help='json paths separated by whitespace')
group.add_argument('--annots_folder', help='path of annotation folder containing multiple jsons')
ap.add_argument('--output_json', help='path of output json', required=True)
args = ap.parse_args()

# create Datumaro project
project = Project()

# add sources
if args.json_paths:
	for i, json_path in enumerate(args.json_paths):
		new_json_path = check_json_path(json_path)
		project.add_source(f'src{i}', {'url': str(new_json_path), 'format': 'coco_instances'})
elif args.annots_folder:
	# doesnt recursively search in subfolders
	for i, json_path in enumerate(Path(args.annots_folder).iterdir()):
		if json_path.suffix == '.json':
			new_json_path = check_json_path(json_path)
			project.add_source(f'src{i}', {'url': str(new_json_path), 'format': 'coco_instances'})

# create a dataset
dataset = project.make_dataset()

# print some stats
print(f'num images: {num_img(dataset)}')
print(f'num images with annotations: {num_img_with_annots(dataset)}')
print(f'num annotations: {num_annots(dataset)}')

# export the resulting json in COCO format
export_json(dataset, args.output_json)
