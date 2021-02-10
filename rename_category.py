import argparse

from datumaro.components.project import Project
from datum_utils import export_json


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

# CHANGE DATASET LABELS HERE (can also be used to merge categories)
dataset = dataset.transform('remap_labels',
							{'Pedestrian': 'person'
							}, default='keep')

# export the resulting dataset in COCO format
export_json(dataset, args.output_json, args.json_path)
