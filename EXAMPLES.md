## Create a dataset from other datasets

```python
dataset = Dataset.from_extractors(dataset1, dataset2)
```

## Change dataset labels

- `default='keep'` to keep all other categories
- `default='delete'` to remove all other categories

```python
dataset = dataset.transform('remap_labels',
    {'pedestrian': 'person', # rename pedestrian to person
    'dog': '' # remove this label
    }, default='keep')
```

## Iterate over dataset elements

```python
for item in dataset:
    print(item.id, item.annotations)
```

## Filter dataset

### Keep only annotated images

```python
dataset = dataset.select(lambda item: len(item.annotations) != 0)
```

## Filter annotations

- `filter_annotations` to filter annotations instead of dataset
- `remove_empty` to remove images without annotations
- [XPath](https://devhints.io/xpath) is used as a query format

### Filter out occluded detections

```python
dataset1 = dataset.filter('/item/annotation[occluded="False"]', filter_annotations=True)
```

### Filter by category label

```python
dataset1 = dataset.filter('/item/annotation[label="person"]', filter_annotations=True)
```

### Filter by bbox width

```python
dataset1 = dataset.filter('/item/annotation[w>5]', filter_annotations=True)
```

## Get dataset statistics

```python
from datum_utils import print_extractor_info, compute_image_statistics, compute_ann_statistics, num_img, num_img_with_annots, num_annots

print(f'num images: {num_img(dataset)}')
print(f'num images with annotations: {num_img_with_annots(dataset)}')
print(f'num annotations: {num_annots(dataset)}')
print_extractor_info(dataset, indent=" ")
stats = {}
stats.update(compute_image_statistics(dataset))
stats.update(compute_ann_statistics(dataset))
pprint(stats)
```

## Reference

- [Datumaro API and developer manual](https://github.com/openvinotoolkit/datumaro/blob/develop/docs/developer_guide.md)
- [Datumaro user manual (CLI)](https://github.com/openvinotoolkit/datumaro/blob/develop/docs/user_manual.md)
