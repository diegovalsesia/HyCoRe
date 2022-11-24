#!/bin/bash

# Train PointMLP + HyCoRe
python main_pointmlp_hycore.py --learning_rate 0.1 --epoch=300 --msg=Offv_pointmlp_hycore --workers=8

# Train PointMLP
# python main_pointmlp.py --learning_rate 0.1 --epoch=300 --msg=Offv_pointmlp --workers=8