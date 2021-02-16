import logging as log
import numpy as np
import shutil
from copy import deepcopy
from pathlib import Path

from datumaro.components.extractor import AnnotationType


def num_img(dataset):
    return len(dataset)

def num_img_with_annots(dataset):
    return len(dataset.select(lambda item: len(item.annotations) != 0))

def num_annots(dataset):
    return sum([len(item.annotations) for item in dataset])

def export_json(dataset, output_path, input_path=""):
    try:
        temp_dir = 'datum_temp'
        dataset.export(temp_dir, 'coco_instances', reindex=True)

        json_glob = list(Path(f'{temp_dir}/annotations').glob('*.json'))
        assert len(json_glob) == 1, "Multiple output json files created! Something is wrong... Rename the input jsons with 'datum_utils.check_json_path'."
        instances_json_path = json_glob[0]

        if len(output_path):
            output_pathlib = Path(output_path)
            if output_pathlib.is_file():
                if input(f'Overwrite {output_pathlib.name}? y/n: ') == 'n':
                    output_path = str(output_pathlib.parent/'datum_split'/output_pathlib.name)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(instances_json_path, output_path)
        else:
            shutil.copyfile(instances_json_path, input_path)
    finally:
        _clear_datum_temps()

def check_json_path(json_path):
    # having multiple jsons with '_' in filename will cause errors
    if '_' in Path(json_path).stem:
        temp_dir = Path('datum_temp_rename')
        temp_dir.mkdir(parents=True, exist_ok=True)
        new_json_path = temp_dir/f"json{len(list(temp_dir.glob('*')))}.json"
        shutil.copyfile(json_path, new_json_path)
        return new_json_path
    else:
        return json_path

def _clear_datum_temps():
    for temp_dir in Path.cwd().glob('datum_temp*'):
        if temp_dir.is_dir():
            shutil.rmtree(temp_dir)

def print_extractor_info(extractor, indent=''):
    print("%slength:" % indent, len(extractor))

    categories = extractor.categories()
    print("%scategories:" % indent, ', '.join(c.name for c in categories))

    for cat_type, cat in categories.items():
        print("%s  %s:" % (indent, cat_type.name))
        if cat_type == AnnotationType.label:
            print("%s    count:" % indent, len(cat.items))

            count_threshold = len(cat.items)
            labels = ', '.join(c.name for c in cat.items[:count_threshold])
            if count_threshold < len(cat.items):
                labels += " (and %s more)" % (
                    len(cat.items) - count_threshold)
            print("%s    labels:" % indent, labels)

def compute_image_statistics(dataset):
    stats = {
        'dataset': {},
        'subsets': {}
    }

    def _extractor_stats(extractor):
        available = True
        for item in extractor:
            if not (item.has_image and item.image.has_data):
                available = False
                log.warn("Item %s has no image. Image stats won't be computed",
                    item.id)
                break

        stats = {
            'images count': len(extractor),
        }

        if available:
            mean, std = mean_std(extractor)
            stats.update({
                'image mean': [float(n) for n in mean[::-1]],
                'image std': [float(n) for n in std[::-1]],
            })
        else:
            stats.update({
                'image mean': 'n/a',
                'image std': 'n/a',
            })
        return stats

    stats['dataset'].update(_extractor_stats(dataset))

    subsets = dataset.subsets() or [None]
    if subsets and 0 < len([s for s in subsets if s]):
        for subset_name in subsets:
            stats['subsets'][subset_name] = _extractor_stats(
                dataset.get_subset(subset_name))

    return stats

def _find_unique_images(dataset, item_hash=None):
    def _default_hash(item):
        if not item.image or not item.image.has_data:
            if item.image and item.image.path:
                return hash(item.image.path)

            log.warning("Item (%s, %s) has no image "
                "info, counted as unique", item.id, item.subset)
            return None
        return hashlib.md5(item.image.data.tobytes()).hexdigest()

    if item_hash is None:
        item_hash = _default_hash

    unique = {}
    for item in dataset:
        h = item_hash(item)
        if h is None:
            h = str(id(item)) # anything unique
        unique.setdefault(h, set()).add((item.id, item.subset))
    return unique

def compute_ann_statistics(dataset):
    labels = dataset.categories().get(AnnotationType.label)
    def get_label(ann):
        return labels.items[ann.label].name if ann.label is not None else None

    unique_images = _find_unique_images(dataset)
    repeated_images = [sorted(g) for g in unique_images.values() if 1 < len(g)]
    unannotated_images = []

    stats = {
        'images count': len(dataset),
        'unique images count': len(unique_images),
        'repeated images count': len(repeated_images),
        # 'repeated images': repeated_images, # [[id1, id2], [id3, id4, id5], ...]
        'annotations count': 0,
        'unannotated images count': 0,
        # 'unannotated images': [],
        'annotations by type': { t.name: {
            'count': 0,
        } for t in AnnotationType },
        'annotations': {},
    }
    by_type = stats['annotations by type']

    attr_template = {
        'count': 0,
        'values count': 0,
        'values present': set(),
        'distribution': {}, # value -> (count, total%)
    }
    label_stat = {
        'count': 0,
        'distribution': { l.name: [0, 0] for l in labels.items
        }, # label -> (count, total%)

        'attributes': {},
    }
    stats['annotations']['labels'] = label_stat
    segm_stat = {
        'avg. area': 0,
        'area distribution': [], # a histogram with 10 bins
        # (min, min+10%), ..., (min+90%, max) -> (count, total%)

        'pixel distribution': { l.name: [0, 0] for l in labels.items
        }, # label -> (count, total%)
    }
    stats['annotations']['segments'] = segm_stat
    segm_areas = []
    pixel_dist = segm_stat['pixel distribution']
    total_pixels = 0

    for item in dataset:
        if len(item.annotations) == 0:
            unannotated_images.append(item.id)
            continue

        for ann in item.annotations:
            by_type[ann.type.name]['count'] += 1

            if not hasattr(ann, 'label') or ann.label is None:
                continue

            if ann.type in {AnnotationType.mask,
                    AnnotationType.polygon, AnnotationType.bbox}:
                area = ann.get_area()
                segm_areas.append(area)
                pixel_dist[get_label(ann)][0] += int(area)

            label_stat['count'] += 1
            label_stat['distribution'][get_label(ann)][0] += 1

            for name, value in ann.attributes.items():
                if name.lower() in { 'occluded', 'visibility', 'score',
                        'id', 'track_id' }:
                    continue
                attrs_stat = label_stat['attributes'].setdefault(name,
                    deepcopy(attr_template))
                attrs_stat['count'] += 1
                attrs_stat['values present'].add(str(value))
                attrs_stat['distribution'] \
                    .setdefault(str(value), [0, 0])[0] += 1

    stats['annotations count'] = sum(t['count'] for t in
        stats['annotations by type'].values())
    stats['unannotated images count'] = len(unannotated_images)

    for label_info in label_stat['distribution'].values():
        label_info[1] = label_info[0] / (label_stat['count'] or 1)

    for label_attr in label_stat['attributes'].values():
        label_attr['values count'] = len(label_attr['values present'])
        label_attr['values present'] = sorted(label_attr['values present'])
        for attr_info in label_attr['distribution'].values():
            attr_info[1] = attr_info[0] / (label_attr['count'] or 1)

    # numpy.sum might be faster, but could overflow with large datasets.
    # Python's int can transparently mutate to be of indefinite precision (long)
    total_pixels = sum(int(a) for a in segm_areas)

    segm_stat['avg. area'] = total_pixels / (len(segm_areas) or 1.0)

    for label_info in segm_stat['pixel distribution'].values():
        label_info[1] = label_info[0] / (total_pixels or 1)

    if len(segm_areas) != 0:
        hist, bins = np.histogram(segm_areas)
        segm_stat['area distribution'] = [{
            'min': float(bin_min), 'max': float(bin_max),
            'count': int(c), 'percent': int(c) / len(segm_areas)
        } for c, (bin_min, bin_max) in zip(hist, zip(bins[:-1], bins[1:]))]

    return stats
