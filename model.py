"""Implements our LASSO model, and loads it from disk."""

import os

import numpy


def makeFullFeatureVector(im, selected_genes, inhibitors, lab_id):
  """Returns a full feature vector, concatenating all components.

  This includes:
   - RNA seq
   - drug response
   - clinical categorical data
   - clinical numerical data
  """
  return numpy.concatenate([
      im.getRnaFeatures(lab_id, selected_genes=selected_genes),
      im.getAucFeatures(lab_id, inhibitors),
      im.getClinicalCategoricalFeatures(lab_id, one_hot=True),
      im.getClinicalNumericalFeatures(lab_id)])


class Model(object):
  """A class which loads and applies a linear hazard model."""
  def __init__(self, model_dir):
    self.model_dir = model_dir


  def load(self):
    """Load state from model_dir."""
    def fromFile(fname):
      return numpy.load(
          os.path.join(self.model_dir, fname + '.npy'), allow_pickle=True)
    self.feature_means = fromFile('feature_means')
    self.feature_stds = fromFile('feature_stds')
    self.estimator_coef = fromFile('estimator_coef')
    self.most_variant_genes = fromFile('most_variant_genes')
    self.inhibitors = fromFile('inhibitors')


  def predictSurvival(self, im, lab_id):
    """Returns the negative hazard predicted by our estimator."""
    feature_vector = makeFullFeatureVector(
        im,
        self.most_variant_genes.tolist(),
        self.inhibitors.tolist(),
        lab_id)
    normed = (feature_vector - self.feature_means) / self.feature_stds
    return -numpy.nan_to_num(normed).dot(self.estimator_coef)

