#!/bin/bash
echo "-----------------------------------------------------"
echo "Emile Male Pipeline: Training"
rsync -az hpc3:/mnt/rds/redhen/gallina/home/rpm93/RedHenLab_Multimodal_Christian_Art_Tagging/ /tmp/$USER/
echo "Files synced successfully"

echo "Creating singularity enviroment...."
module load singularity/3.8.1
singularity pull docker://ghcr.io/lazycodes7/christian-art-tagging:latest
mkdir curation/EmileMaleDataset
echo "Enviroment successfully created."

echo "Stage: 1 - Generating the Curated Dataset"
singularity run --nv art-detector-yolov6_latest.sif python curation/generator.py --metadata_path curation/metadata_v2.csv --data_dir curation/EmileMaleDataset/

echo "Stage: 2- Training the feature-extractor to extract patch level features"
singularity run --nv art-detector-yolov6_latest.sif python feature_extractor/train.py -c --train --device cuda --data_dir feature_extractor/

echo "Stage: 3- Training the Image-Captioning model that uses intra-modal features"
singularity run --nv art-detector-yolov6_latest.sif python captioning/train.py --data_dir curation/EmileMaleDataset/ --feature_extractor_path feature_extractor/artDL.pt --device cuda --train
