
import logging
import itertools

from .multichoice_align import heuristic_align
from .multichoice_align import model_judge_align, model_judge_align_batch

from tqdm import tqdm

def cross_lingual_assessment(data_with_model_prediction):
    """ Compute the score of the model on the given data."""

    evaluated_languages = []
    for sample_set in data_with_model_prediction:
        for sample in sample_set:
            if sample == 'id':
                continue
            if sample not in evaluated_languages:
                evaluated_languages.append(sample)

    logging.info(f'evaluated_languages: {evaluated_languages}')

    # Add align prediction to data_with_model_prediction
    for sample_set in tqdm(data_with_model_prediction):


        choices_list                = [sample_set[sample]['choices'] for sample in sample_set if sample != 'id']
        model_prediction_list       = [sample_set[sample]['model_prediction'] for sample in sample_set if sample != 'id']
        model_prediction_align_list = model_judge_align_batch(choices_list, model_prediction_list)

        for sample in sample_set:
            if sample == 'id':
                continue
            sample_set[sample]['model_prediction_align'] = model_prediction_align_list.pop(0)

            # sample_set[sample]['model_prediction_align'] = heuristic_align(sample_set[sample]['choices'], sample_set[sample]['model_prediction'])
            #sample_set[sample]['model_prediction_align'] = model_judge_align(sample_set[sample]['choices'], sample_set[sample]['model_prediction'])

    # Check if the model prediction is with in the choices
    for sample_set in data_with_model_prediction:
        for sample in sample_set:
            if sample == 'id':
                continue
            if sample_set[sample]['model_prediction_align'] not in sample_set[sample]['choices']:
                logging.warning(f'Model prediction align {sample_set[sample]["model_prediction_align"]} is not in the choices {sample_set[sample]["choices"]}')

    results  = {}
    lang2acc = {}

    # Compute the score for each language
    for language in evaluated_languages:
        sample_true_false = []
        for sample_set in data_with_model_prediction:
            if sample_set[language]['model_prediction_align'] == sample_set[language]['answer']:
                sample_true_false.append(1)
            else:
                sample_true_false.append(0)
        lang2acc[language] = sum(sample_true_false) / len(sample_true_false)
    
    results['overall_acc']  = sum(lang2acc.values()) / len(lang2acc.values())
    results['language_acc'] = lang2acc

    # Compute the consistency score
    consistency_scores = {}
    for i in range(2, len(evaluated_languages)+1):
        combinations = itertools.combinations(evaluated_languages, i)

        combination_consistency_scores = {}
        for combination in combinations:
            current_consistency_score = []

            for sample_set in data_with_model_prediction:
                answers = [sample_set[lang]['model_prediction_align'][0:3] for lang in combination]
                if len(set(answers)) == 1:
                    current_consistency_score.append(1)
                else:
                    current_consistency_score.append(0)
            
            combination_consistency_scores[','.join(combination)] = sum(current_consistency_score) / len(current_consistency_score)

        consistency_scores['{}_combine'.format(i)] = combination_consistency_scores

    for i in range(2, len(evaluated_languages)+1):
        results[f'consistency_score_{i}'] = sum(consistency_scores['{}_combine'.format(i)].values()) / len(consistency_scores['{}_combine'.format(i)].values())

    results['detailed_consistency_score'] = consistency_scores

    # Compute the AC3 score
    for i in range(2, len(evaluated_languages)+1):
        consistency_i = results[f'consistency_score_{i}']
        results[f'AC3_{i}'] = 2 * consistency_i * results['overall_acc'] / (consistency_i + results['overall_acc'] + 1e-10)

    return results, data_with_model_prediction