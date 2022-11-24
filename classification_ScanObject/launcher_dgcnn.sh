#!/bin/bash

# Train DGCNN + HyCoRe
python main_dgcnn_hycore.py --learning_rate 0.1 --epoch=250 --msg=Offv_dgcnn_hycore --workers=8

# Train DGCNN
# python main_dgcnn.py --learning_rate 0.1 --epoch=250 --msg=Offv_dgcnn --workers=8