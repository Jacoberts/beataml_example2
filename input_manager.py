"""Classes for handling the input data files."""

import os

import numpy
import pandas
from sklearn.preprocessing import OneHotEncoder


class RawInputs(object):
  """Holds raw input files.

  This is separated from InputManager so the latter can be recreated / reloaded
  more frequently during development."""
  def __init__(self, training_dir):
    self.training_dir = training_dir

  def load(self):
    def getCsv(fname):
      print(f'Loading {fname} data...', flush=True)
      return pandas.read_csv(os.path.join(self.training_dir, fname + '.csv'))
    self.rnaseq = getCsv('rnaseq')
    self.dnaseq = getCsv('dnaseq')
    self.aucs = getCsv('aucs')
    self.clinical_numerical = getCsv('clinical_numerical')
    self.clinical_categorical = getCsv('clinical_categorical')
    self.clinical_categorical_legend = getCsv('clinical_categorical_legend')


class InputManager(object):
  """Manages training data inputs."""
  def __init__(self, raw_inputs):
    self.raw = raw_inputs


  def prepInputs(self):
    """Prepare some tmp variables so accessing later is quick."""
    # Reindex some raw fields for quick accession.
    self.clinical_categorical = (
        self.raw.clinical_categorical.set_index('lab_id'))
    self.clinical_numerical = (self.raw.clinical_numerical
        .set_index('lab_id')
        .astype('float64'))

    # Slice RNAseq by specimen, and normalize.
    self.rnaseq_by_spec = self.raw.rnaseq
    self.rnaseq_by_spec.index = self.rnaseq_by_spec.Gene
    self.rnaseq_by_spec = self.rnaseq_by_spec[self.rnaseq_by_spec.columns[2:]].T
    self.rnaseq_by_spec = self.rnaseq_by_spec.apply(
        lambda spec: spec / numpy.linalg.norm(spec), axis=1)

    # Create a feature matrix of drug AUC z-score.
    self.aucs = self.raw.aucs.pivot(
        index='lab_id', columns='inhibitor', values='auc')

    # Create a OneHotEncoder for categorical data. We use the legend to get the
    # list of all enum values per categorical field, which we pass into the
    # encoder. Then we "fit" the encoder, which just verifies that all
    # categories in the data are represented in the legend (sure hope so!).
    enums_per_field = (
        self.raw.clinical_categorical_legend.groupby('column').enum.apply(
          lambda values: sorted(list(values))))
    enums_per_field = enums_per_field[self.clinical_categorical.columns]
    self.clinical_categorical_encoder = OneHotEncoder(
        categories=enums_per_field.tolist(), sparse=False)
    self.clinical_categorical_encoder.fit(self.clinical_categorical)
  

  def printStats(self):
    print(f'''Found:
    {self.raw.aucs.lab_id.unique().shape[0]} unique specimens
    {self.raw.aucs.inhibitor.unique().shape[0]} unique inhibitors
    {self.raw.clinical_categorical.columns.shape[0]} clinical categorical fields''')


  def getAllSpecimens(self):
    """Returns a list of all specimens. No guarantees on ordering!"""
    return self.raw.aucs.lab_id.unique().tolist()


  def getRnaFeatures(self, lab_id, selected_genes=None):
    """Returns ndarray of all RNA gene feature counts, normalized log2(cpm)."""
    row = self.rnaseq_by_spec.loc[lab_id]
    if selected_genes is not None:
      row = row[selected_genes]
    return row.to_numpy()


  def getAucFeatures(self, lab_id):
    """Returns ndarray of response to inhibitors, with NaNs."""
    return self.aucs.loc[lab_id].to_numpy()
    

  def getClinicalCategoricalFeatures(self, lab_id, one_hot=False):
    """Return clinical category feature ndarray, optionally one-hot encoded."""
    raw_enums = self.clinical_categorical.loc[lab_id].to_numpy()
    if not one_hot:
      return raw_enums
    return self.clinical_categorical_encoder.transform(
        raw_enums.reshape(1, -1))[0]


  def getClinicalNumericalFeatures(self, lab_id):
    """Return ndarray of numerical clinical fields."""
    return self.clinical_numerical.loc[lab_id].to_numpy()
