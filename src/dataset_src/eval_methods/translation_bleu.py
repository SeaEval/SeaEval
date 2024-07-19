
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def translation_bleu(data_with_model_prediction):
    """ Compute the score of the model on the given data."""

    bleu_smooth_function = SmoothingFunction()
    #sentence_bleu = sentence_bleu


    bleu_sentence_scores = []

    for sample in data_with_model_prediction:
        reference_sent = word_tokenize(sample['answer'])
        model_generation = word_tokenize(sample['model_prediction'])

        bleu_sentence_scores.append(sentence_bleu(hypothesis=model_generation, references=[reference_sent], smoothing_function=bleu_smooth_function.method1))

    bleu_score = sum(bleu_sentence_scores) / len(bleu_sentence_scores)
    results = {'bleu_score': bleu_score}

    return results, data_with_model_prediction