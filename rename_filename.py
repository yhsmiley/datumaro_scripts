import argparse
from pathlib import Path

from datumaro.components.project import Project
from datum_utils import export_json


ap = argparse.ArgumentParser()
ap.add_argument('--json_path', help='json path', required=True)
ap.add_argument('--output_json', default='', help='path of output json. overwrite input json if not specified')
args = ap.parse_args()

# create Datumaro project
project = Project()

# add sources
project.add_source('src1', {'url': args.json_path, 'format': 'coco_instances'})

# create a dataset
dataset = project.make_dataset()

for item in dataset:
	# WRITE YOUR RENAME FUNCTION HERE
	item.id = item.id.split('/')[-1] # only keep filename

# export the resulting dataset in COCO format
export_json(dataset, args.output_json, args.json_path)
