
import os
import random
from tqdm import tqdm
import transformers
from openai import OpenAI



def model_judge(question, reference, prediction):

    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", device_map="auto", use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Judgement model
    # port = random.choice([os.environ.get('VLLM_PORT', 5000)])
    port = os.environ.get('MY_VLLM_PORT_JUDGE', 5000)

    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    model = models.data[0].id
    
    # Generation
    prediction = prediction.strip()

    PROMPT_TEMPLATE = """\
        [Reference Answer]
        {reference}

        [Model Answer]
        {prediction}

        [Question]
        {question}

        [Task]
        Rate the model's answer based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details.
        Criteria: Assess if the model's response mirrors the reference in terms of content, accuracy, and relevance.
        Score0: The answer is completely misaligned, providing incorrect or irrelevant information compared to the reference.
        Score1: The answer shows minimal alignment, often misunderstanding or providing irrelevant details unrelated to the reference.
        Score2: The answer recognizes the topic but diverges significantly from the reference in accuracy or relevance.
        Score3: The answer aligns with the reference generally but lacks detail or precise accuracy in some aspects.
        Score4: The answer is mostly accurate and relevant, closely following the reference but could be clearer or more detailed.
        Score5: The answer is highly accurate, detailed, and matches the reference answer perfectly, capturing its essence and detail.

        Format your response as follows, and nothing after the rating number:
        Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
        Rating: (int)"""

    evaluation_prompt = PROMPT_TEMPLATE.format(reference=reference, prediction=prediction, question=question)
    messages = [
        {"role": "user", "content": evaluation_prompt},
    ]

    templated_sample = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=False,
    )

    completion = client.completions.create(
    model      = model,
    prompt     = templated_sample,
    max_tokens = 512,
    n          = 1,
        )
    
    output = completion.choices[0].text.strip()

    try:
        justification = output.split("Explanation:")[1].split("Rating:")[0].strip()
        score         = float(output.split()[-1])
        success       = 1
    except:
        breakpoint()
        justification = ""
        score         = 0.0
        success       = 0

    return justification, score
        

def open_ended_question(data_with_model_prediction, category):
    """ Compute the score of the model on the given data."""

    scores = []
    for sample in tqdm(data_with_model_prediction):

        justification, score                = model_judge(sample['question'], sample['post_edited_answer'], sample['model_prediction'])
        sample["model_judge_justification"] = justification
        sample["model_judge_score"]         = score
        scores.append(score)

    average_score = sum(scores) / len(scores)
    results = {'model_judge_score': average_score*20}

    return results, data_with_model_prediction