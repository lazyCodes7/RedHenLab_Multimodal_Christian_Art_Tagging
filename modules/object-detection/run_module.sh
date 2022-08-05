#!/bin/bash
rsync -az hpc3:/mnt/rds/redhen/gallina/home/rpm93/RedHenLab_Multimodal_Christian_Art_Tagging/ /tmp/$USER/
cd rpm93/modules/object-detection/YOLOv6
module load singularity/3.8.1
singularity pull docker://lazycodes7/art-detector-yolov6:latest
singularity run --nv art-detector-yolov6_latest.sif python tools/train.py --batch 32 --conf configs/yolov6s.py --data Iconart/data.yaml --device 0 --workers 0