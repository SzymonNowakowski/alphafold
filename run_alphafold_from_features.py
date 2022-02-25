# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import sys
import time
from typing import Dict

sys.path.insert(0, '/alphafold_current')

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax
import numpy as np

from alphafold.model import data
# Internal import (7716).

logging.set_verbosity(logging.INFO)

flags.DEFINE_list(
    'features_paths', None, 'Paths to features pickle files, containing '
    'precomputed features. Paths should be separated by commas. '
    'The parent directory is used as sequence name and is used to '
    'and is used to name the output directories for each prediction.')

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_string('structural_hypothesis_file', None,
                    'Auxiliary structural hypothesis PDB file to warm-start alphafold iterations')

FLAGS = flags.FLAGS

RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def predict_structure_from_features(
    features_path: str,
    fasta_name: str,
    output_dir_base: str,
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int):

  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  output_dir = os.path.join(output_dir_base, fasta_name)

  timings = {}

  # Load out features from a pickled dictionary.
  with open(features_path, 'rb') as f:
    feature_dict = pickle.load(f)

  unrelaxed_pdbs = {}
  relaxed_pdbs = {}
  ranking_confidences = {}

  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    timings[f'process_features_{model_name}'] = time.time() - t_0

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict,
                                             random_seed=model_random_seed)
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, t_diff)

    if benchmark:
      t_0 = time.time()
      model_runner.predict(processed_feature_dict,
                           random_seed=model_random_seed)
      t_diff = time.time() - t_0
      timings[f'predict_benchmark_{model_name}'] = t_diff
      logging.info(
          'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
          model_name, fasta_name, t_diff)

    plddt = prediction_result['plddt']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(prediction_result, f, protocol=4)

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)

    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])

    if amber_relaxer:
      # Relax the prediction.
      t_0 = time.time()
      relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
      timings[f'relax_{model_name}'] = time.time() - t_0

      relaxed_pdbs[model_name] = relaxed_pdb_str

      # Save the relaxed PDB.
      relaxed_output_path = os.path.join(
          output_dir, f'relaxed_{model_name}.pdb')
      with open(relaxed_output_path, 'w') as f:
        f.write(relaxed_pdb_str)

  # Rank by model confidence and write out relaxed PDBs in rank order.
  ranked_order = []
  for idx, (model_name, _) in enumerate(
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
    ranked_order.append(model_name)
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      if amber_relaxer:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))

  logging.info('Final timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings_pred.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run_multimer_system = 'multimer' in FLAGS.model_preset

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).parent.stem for p in FLAGS.features_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All features paths must have a unique parent directory.')

  model_runners = {}
  model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  for model_name in model_names:
    model_config = config.model_config(model_name)
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_config.model.structural_hypothesis_file = FLAGS.structural_hypothesis_file
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_names))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  for i, features_path in enumerate(FLAGS.features_paths):
    fasta_name = fasta_names[i]
    predict_structure_from_features(
        features_path=features_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'features_paths',
      'output_dir',
      'data_dir',
  ])

  app.run(main)
