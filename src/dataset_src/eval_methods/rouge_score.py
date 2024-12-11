from rouge_score import rouge_scorer


def rouge_score(data_with_model_prediction):
    """ Compute the score of the model on the given data."""

    rouge_scorer_model = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, split_summaries=True)

    rouge1 = 0
    rouge2 = 0
    rougeL = 0

    for sample in data_with_model_prediction:
        scores = rouge_scorer_model.score(sample['answer'], sample['model_prediction'])
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure

    rouge1 /= len(data_with_model_prediction)
    rouge2 /= len(data_with_model_prediction)
    rougeL /= len(data_with_model_prediction)

    avg_rouge = (rouge1 + rouge2 + rougeL) / 3

    results = {
        'rouge1'   : rouge1,
        'rouge2'   : rouge2,
        'rougeL'   : rougeL,
        'avg_rouge': avg_rouge,
    }

    return results, None
