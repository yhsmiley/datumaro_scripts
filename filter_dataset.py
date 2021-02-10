import argparse
from pathlib import Path

from datumaro.components.project import Project
from datum_utils import num_img, num_img_with_annots, num_annots, export_json


ap = argparse.ArgumentParser()
ap.add_argument('--json_path', help='annotations json path', required=True)
ap.add_argument('--output_json', default='', help='path of output json. overwrite input json if not specified')
args = ap.parse_args()

# create Datumaro project
project = Project()

# add sources
project.add_source('src1', {'url': args.json_path, 'format': 'coco_instances'})

# create a dataset
dataset = project.make_dataset()

# print original stats
print('original stats')
print(f'num images: {num_img(dataset)}')
print(f'num images with annotations: {num_img_with_annots(dataset)}')
print(f'num annotations: {num_annots(dataset)}')

# WRITE YOUR FILTER HERE
dataset1 = dataset.filter('/item/annotation[w>5]', filter_annotations=True)

# print filtered stats
print('filtered stats')
print(f'num images: {num_img(dataset1)}')
print(f'num images with annotations: {num_img_with_annots(dataset1)}')
print(f'num annotations: {num_annots(dataset1)}')

# export the resulting dataset in COCO format
export_json(dataset1, args.output_json, args.json_path)
