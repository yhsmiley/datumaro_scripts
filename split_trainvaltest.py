import argparse
from pathlib import Path

from datumaro.components.project import Project
from datum_utils import num_img, num_img_with_annots, num_annots, export_json


ap = argparse.ArgumentParser()
ap.add_argument('--json_path', help='annotations json path', required=True)
args = ap.parse_args()

# WRITE YOUR SPLIT HERE
splits = {'train': 
				['set00', 'set01'], 
		  'val': 
		  		['set02'], 
		  'test': 
		  		['set03']}

# create Datumaro project
project = Project()

# add sources
project.add_source('src1', {'url': args.json_path, 'format': 'coco_instances'})

# create a dataset
dataset = project.make_dataset()
print(f'total images: {num_img(dataset)}')

for split_name, split_list in splits.items():
	# DEFINE SPLIT FUNCTION HERE
	dataset_split = dataset.select(lambda item: item.id.startswith(tuple(split_list)))

	# print some stats
	print(f'split: {split_name}')
	print(f'num images: {num_img(dataset_split)}')
	print(f'num images with annotations: {num_img_with_annots(dataset_split)}')
	print(f'num annotations: {num_annots(dataset_split)}')

	# export the resulting dataset in COCO format
	output_json_path = str(Path(args.json_path).parent/Path(split_name+'.json'))
	export_json(dataset_split, output_json_path)
