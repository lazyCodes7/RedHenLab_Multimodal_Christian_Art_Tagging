#!/bin/bash
rsync -az hpc3:/mnt/rds/redhen/gallina/home/rpm93/RedHenLab_Multimodal_Christian_Art_Tagging/ /tmp/$USER/
cd rpm93/pipeline/
module load singularity/3.8.1
singularity pull docker://lazycodes7/christian-art-tagging:latest
mkdir curation/EmileMaleDataset
singularity run --nv christian-art-tagging_latest.sif python curation/generator.py --metadata_path curation/metadata_v2.csv --data_dir curation/EmileMaleDataset/
singularity run --nv christian-art-tagging_latest.sif python feature_extractor/train.py -c --train --device cuda --data_dir feature_extractor/
