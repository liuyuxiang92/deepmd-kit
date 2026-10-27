# DPA4 classification example

Run `python make_data.py` to generate the synthetic two-class dataset, then
`dp --pt train input_dpa4.json` or `dp --pt-expt train input_dpa4.json`.
Use separate directories for the two runs to keep their checkpoints apart.

Each frame has one scalar label in `class.npy`: class 0 has a bond length near
0.75 Å and class 1 near 1.5 Å. These are toy categories, not physical phases.
See [the classification guide](../../../doc/model/train-classification.md) for
the data contract, configuration, and prediction API.
