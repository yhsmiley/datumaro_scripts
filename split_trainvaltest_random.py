import argparse
from pathlib import Path

from datumaro.components.project import Project
from datum_utils import num_img, num_img_with_annots, num_annots, export_json


ap = argparse.ArgumentParser()
ap.add_argument('--json_path', help='annotations json path', required=True)
ap.add_argument('--split_ratio', nargs="+", type=float, default=[8, 1, 1], help='ratio of train/val or train/val/test split separated by whitespace. default is 8:1:1')
args = ap.parse_args()

# create Datumaro project
project = Project()

# add sources
project.add_source('src1', {'url': args.json_path, 'format': 'coco_instances'})

# create a dataset
dataset = project.make_dataset()
print(f'total images: {num_img(dataset)}')

split_ratio = args.split_ratio
train_ratio = split_ratio[0] / sum(split_ratio)
val_ratio = split_ratio[1] / sum(split_ratio)
test_ratio = split_ratio[2] / sum(split_ratio)
print(f'train/val/test ratio: {train_ratio} {val_ratio} {test_ratio}')

dataset = dataset.transform('random_split',
			[('train', train_ratio),
			('val', val_ratio),
			('test', test_ratio)])

for subset_name, subset in dataset.subsets().items():
	# print some stats
	print(f'subset: {subset_name}')
	print(f'num images: {num_img(subset)}')
	print(f'num images with annotations: {num_img_with_annots(subset)}')
	print(f'num annotations: {num_annots(subset)}')

	# export the resulting dataset in COCO format
	subset_to_export = dataset.select(lambda item: item.subset==subset_name)
	output_json_path = str(Path(args.json_path).parent/Path(subset_name+'.json'))
	export_json(subset_to_export, output_json_path)
