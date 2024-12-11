
from .multichoice_align import heuristic_align
from .multichoice_align import model_judge_align, model_judge_align_batch

from tqdm import tqdm

def multichoice_question(data_with_model_prediction, category):
    """ Compute the score of the model on the given data."""

    # for sample in data_with_model_prediction:
    #   sample['model_prediction_align'] = heuristic_align(sample['choices'], sample['model_prediction'])

    accuracy = []

    choices_list                = [sample['choices'] for sample in data_with_model_prediction]
    model_prediction_list       = [sample['model_prediction'] for sample in data_with_model_prediction]
    model_prediction_align_list = model_judge_align_batch(choices_list, model_prediction_list)

    for sample in tqdm(data_with_model_prediction):
        sample['model_prediction_align'] = model_prediction_align_list.pop(0)
        if sample['model_prediction_align'] == sample['answer']:
            accuracy.append(1)
        else:
            accuracy.append(0)

    accuracy = sum(accuracy) / len(accuracy)
    results = {'accuracy': accuracy}
    
    if category == True:
        category2acc = {}
        for sample in data_with_model_prediction:
            if sample['model_prediction_align'] == sample['answer']:
                if sample['category'] not in category2acc:
                    category2acc[sample['category']] = [1]
                else:
                    category2acc[sample['category']] += [1]
            else:
                if sample['category'] not in category2acc:
                    category2acc[sample['category']] = [0]
                else:
                    category2acc[sample['category']] += [0]

        for category in category2acc:
            category2acc[category] = sum(category2acc[category]) / len(category2acc[category])

        results['category_acc'] = category2acc

    return results, data_with_model_prediction