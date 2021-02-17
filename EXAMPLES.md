## Create a dataset from other datasets

```python
dataset = Dataset.from_extractors(dataset1, dataset2)
```

## Change dataset labels

- `default='keep'` to keep all other categories
- `default='delete'` to remove all other categories
- NOTE: If you want to remove a category, have to specify all other categories and `default='delete'`. Doesnt work if `default='keep'`.

```python
dataset = dataset.transform('remap_labels',
    {'pedestrian': 'person', # rename pedestrian to person
    'dog': '' # remove this label
    }, default='delete')
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

### Keep images with names that start with 'test_string'

```python
dataset = dataset.select(lambda item: item.id.startswith('test_string'))
```

## Filter annotations

- `filter_annotations` to filter annotations instead of dataset items (images)
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

### Filter with custom conditions
```python
for item in dataset:
    for annot in item.annotations:
        if condition:
            annot.attributes["to_remove"] = True

dataset1 = dataset.filter(f"/item/annotation[not(to_remove='True')]", filter_annotations=True)
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
