"""A script which executes an drug repurposing model on input files.

Expects:
  /model/: should have feature_means.csv, features_stds.csv,
      estimator_coef.csv, most_variant_genes.csv.
  /input/: should have all the input files.

Creates:
  /output/predictions.csv: a CSV with columns ['lab_id', 'survival']
"""

import pandas

from input_manager import InputManager
from input_manager import RawInputs
from model import Model


if __name__ == "__main__":
  # Loading input files.
  raw_inputs = RawInputs('/input')
  raw_inputs.load()
  im = InputManager(raw_inputs)
  im.prepInputs()
  im.printStats()

  # Loading model params.
  model = Model('/model')
  model.load()

  lab_ids = im.getAllSpecimens()

  survivals = []
  for lab_id in lab_ids:
    survivals.append(model.predictSurvival(im, lab_id))

  pandas.DataFrame({
    'lab_id': lab_ids,
    'survival': survivals,
  }).to_csv('/output/predictions.csv', index=False)
