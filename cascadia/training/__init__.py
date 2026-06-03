"""Supervised training for Cascadia's per-hazard leaf predictors.

Phase-2 ML: assemble labelled examples automatically from open data and fit a
real classifier, replacing the calibrated-sigmoid heuristic for that hazard.
The flood predictor is the first trained leaf (clean labels from USGS gage
records). See `dataset.py` and `train_flood.py`.
"""
