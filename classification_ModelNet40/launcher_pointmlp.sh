#!/bin/bash

## PointMLP + HyCoRe
python main_pointmlp_hycore.py --seed=4780 --workers=8 --msg=Offv_pointmlp_hycore_var
# with voting
#python voting_pointmlp_hycore.py-- model Hype_PointNet --msg=20220405170724-4780 

## PointMLP (Euclidean)
#python main_pointmlp.py --seed=4780 --workers=8  --msg=Offv_pointmlp



