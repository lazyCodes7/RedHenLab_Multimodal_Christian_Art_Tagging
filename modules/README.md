# Modules
This is the directory that consists of the modules used in the pipeline. Not all the modules I built here were used in the pipeline but we can understand what each one does.

## 1. Curation
The curation module here was used to analyze the metadata collected and is also used by the pipeline to generate the dataset.

## 2. Captioning 
The captioning module consists of the various transformer architectures that are used in the pipeline

## 3. Feature Extraction
The feature_extraction module is primarily focussed on using CNN as a feature extractor. Note that this is not included in the pipeline

## 4. Object-Detection
The object-detection module has two usecases. One for detecting the saints and attributes in the painting and other to acts as a feature extractor. There are two implementations in the module. The first one is FRCNN and the second one is YOLOv6. The current stage of the pipeline does not use either but instead relies on a vision transformer for extracting features
