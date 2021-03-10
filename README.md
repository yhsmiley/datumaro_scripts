# datumaro_scripts

This repo serves as a collection of scripts for using the datumaro python API.

Refer to `EXAMPLES.md` for example usage.

Currently, scripts are used to preprocess COCO jsons.

## TODO

- [ ] upgrade code to support datumaro upgrade from 0.1.5.1 to 0.1.6.1
- [ ] fix "iscrowd" bug (might be fixed with datumaro upgrade)

## Installation

- First, install pycocotools
```
pip3 install cython numpy
pip3 install pycocotools
```
- Now you can install datumaro
```
pip3 install datumaro
```

## Notes

- There will be a `ModuleNotFoundError: No module named 'tensorflow'` warning, but this is not an issue as we are not using tfrecords. You can just install tensorflow to get rid of this warning.

## Dataset class

The `Dataset` class from the `datumaro.components.dataset` module represents a dataset, consisting of multiple `DatasetItem`. Annotations are represented by members of the `datumaro.components.extractor` module, such as `Label`, `Mask` or `Polygon`. A dataset can contain items from one or multiple subsets (e.g. `train`, `test`, `val` etc.), the list of dataset subsets is available at `dataset.subsets`.

Datasets typically have annotations, and these annotations can require additional information to be interpreted correctly. For instance, it can include class names, class hierarchy, keypoint connections, class colors for masks, class attributes. This information is stored in `dataset.categories`, which is a mapping from `AnnotationType` to a corresponding ...`Categories` class. Each annotation type can have its `Categories`. Typically, there will be a `LabelCategories` object. Annotations and other categories adress dataset labels by their indices in this object.

The main operation for a dataset is iteration over its elements. An item corresponds to a single image, a video sequence, etc. There are also few other operations available, such as filtration (`dataset.select`) and transformations (`dataset.transform`). A dataset can be created from extractors or other datasets with `Dataset.from_extractors()` and directly from items with `Dataset.from_iterable()`. A dataset is an extractor itself. If it is created from multiple extractors, their categories must match, and their contents will be merged.

A dataset item is an element of a dataset. Its id is a name of a corresponding image. There can be some image `attributes`, an `image` and `annotations`.

## Reference

- [Dataset Management Framework (Datumaro)](https://github.com/openvinotoolkit/datumaro)
