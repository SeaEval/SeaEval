#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Monday, July 24th 2023, 11:51:25 am
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


DATASET_SPLIT = {
                'cross_mmlu'    : 'cross_mmlu.json',
                'cross_logiqa'  : 'cross_logiqa.json',
                'sg_eval'       : 'sg_eval.json',
                'us_eval'       : 'us_eval.json',
                'cn_eval'       : 'cn_eval.json',
                'sing2eng'      : 'sing2eng.json',
                'flores_ind2eng': 'flores_ind2eng.json',
                'flores_vie2eng': 'flores_vie2eng.json',
                'flores_zho2eng': 'flores_zho2eng.json',
                'flores_zsm2eng': 'flores_zsm2eng.json',
                'mmlu'          : 'mmlu.json',
                'c_eval'        : 'c_eval.json',
                'cmmlu'         : 'cmmlu.json',
                'zbench'        : 'zbench.json',
                'ind_emotion'   : 'ind_emotion.json',
                'ocnli'         : 'ocnli_mc.json',
                'c3'            : 'c3_mc.json',
                'dream'         : 'dream.json',
                'samsum'        : 'samsum.json',
                'dialogsum'     : 'dialogsum.json',
                'sst2'          : 'sst2_mc.json',
                'cola'          : 'cola_mc.json',
                'qqp'           : 'qqp_mc.json',
                'mnli'          : 'mnli_mc.json',
                'qnli'          : 'qnli_mc.json',
                'wnli'          : 'wnli_mc.json',
                'rte'           : 'rte_mc.json',
                'mrpc'          : 'mrpc_mc.json',
            }



DATASET_TYPE = {
                    'cross_mmlu'    : 'cross_mmlu',
                    'cross_logiqa'  : 'cross_logiqa',
                    'sing2eng'      : 'to_eng_translation',
                    'sg_eval'       : 'eng_multi_choice_no_context',
                    'us_eval'       : 'local_eval_us',
                    'cn_eval'       : 'local_eval_cn',
                    'flores_ind2eng': 'to_eng_translation',
                    'flores_vie2eng': 'to_eng_translation',
                    'flores_zho2eng': 'to_eng_translation',
                    'flores_zsm2eng': 'to_eng_translation',
                    'mmlu'          : 'eng_multi_choice_no_context',
                    'c_eval'        : 'chi_multi_choice_no_context',
                    'cmmlu'         : 'chi_multi_choice_no_context',
                    'zbench'        : 'chi_multi_choice_no_context',
                    'ind_emotion'   : 'ind_classification_multi_choice',
                    'ocnli'      : 'chi_ocnli_multi_choice',
                    'c3'         : 'chi_c3_multi_choice',
                    'dream'         : 'eng_multi_choice_context',
                    'samsum'        : 'eng_summarization',
                    'dialogsum'     : 'eng_summarization',
                    'sst2'       : 'eng_classification_multi_choice',
                    'cola'       : 'eng_classification_multi_choice',
                    'qqp'        : 'eng_classification_multi_choice',
                    'mnli'       : 'eng_classification_multi_choice',
                    'qnli'       : 'eng_classification_multi_choice',
                    'wnli'       : 'eng_classification_multi_choice',
                    'rte'        : 'eng_classification_multi_choice',
                    'mrpc'       : 'eng_classification_multi_choice',
            }

PROMPT_TEMPLATE = {

                    'cross_mmlu': [
                        "Respond to the question by selecting the most appropriate answer.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Kindly choose the correct answer from the options provided for the multiple-choice question.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Solve the multi-choice question by selecting the accurate answer.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Please answer the following multiple-choice question by selecting the correct option.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "As an expert, your task is to solve the following multiple-choice question. Identify the correct response among the given choices.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],

                    'cross_logiqa': [
                        "Respond to the question by selecting the most appropriate answer.\n\nContent:\n{}\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Kindly choose the correct answer from the options provided for the multiple-choice question.\nContent:\n{}\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Solve the multi-choice question by selecting the accurate answer.\nContent:\n{}\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Please answer the following multiple-choice question by selecting the correct option.\n\nContent:\n{}\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "As an expert, your task is to solve the following multiple-choice question. Identify the correct response among the given choices.\n\nContent:\n{}\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],

                    'sing2eng': [
                        "Translate the following sentence from Singlish to English. Please only output the translated sentence.\n\nInput:\n{}\n\nOutput:\n",
                        "Translate the following Singlish text to standard English. Please only output the translated sentence.\nInput:\n{}\nOutput:\n",
                        "Given the text in Singlish, translate it to standard English. Please only output the translated sentence.\nInput:\n{}\nOutput:\n",
                        "Given the sentence below, perform machine translation from Singlish to English. Please only output the translated sentence.\n\nInput:\n{}\n\nOutput:\n",
                        "Please translate the sentence: {} from Singapore-style English to standard English. Please only output the translated sentence.\n\nOutput:\n",
                    ],

                    'sg_eval': [
                        "Please carefully read the following question and select the most appropriate answer from the choices.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Read the following question carefully and select the correct answer from the choices.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Please select the most appropriate option to answer the question from your perspective as a resident of Singapore.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Please answer the following Singapore-related questions by selecting the most probable answer from the choices.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "As a person living in Singapore, try your best to answer the question by selecting the most appropriate option.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],

                    'us_eval': [
                        "Please carefully read the following question and select the most appropriate answer from the choices.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Read the following question carefully and select the correct answer from the choices.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Please select the most appropriate option to answer the question from your perspective as a resident of the United States.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Please provide your answers to the following questions related to the United States by selecting the most likely option from the choices provided.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "As a person living in the United States, try your best to answer the question by selecting the most appropriate option.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],

                    'cn_eval': [
                        "请仔细阅读以下问题，并从选项中选择最合适的答案。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "仔细阅读以下问题，并从选项中选择正确答案。\n问题:\n{}\n选项:\n{}\n答案:\n",
                        "请从您作为中国居民的角度，选择最合适的选项来回答问题。\n问题:\n{}\n选项:\n{}\n答案:\n",
                        "请通过从提供的选项中选择最可能的答案，回答与中国文化相关的以下问题。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "作为生活在中国的居民，请尽力选择最合适的选项来回答问题。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                    ],

                    'flores_ind2eng': [
                        "Translate the following sentence from Indonesian to English.\n\nSentence in Indonesian:\n{}\n\nTranslation in English:\n",
                        "Please translate the provided Indonesian text into English. Output the translated content only.\nSentence in Indonesian:\n{}\nTranslation in English:\n",
                        "Translate the Indonesian text provided into English and provide only the translated content.\nSentence in Indonesian:\n{}\nTranslation in English:\n",
                        "Given the sentence below, perform machine translation from Indonesian to English. Output the translated content only.\n\nSentence in Indonesian:\n{}\n\nTranslation in English:\n",
                        "Please translate the sentence: \"{}\" from Indonesian to English. Output the translated content only.\n\nTranslation in English:\n",
                    ],
                    
                    'flores_vie2eng': [
                        "Translate the following sentence from Vietnamese to English.\n\nSentence in Vietnamese:\n{}\n\nTranslation in English:\n",
                        "Please translate the provided Vietnamese text into English. Output the translated content only.\nSentence in Vietnamese:\n{}\nTranslation in English:\n",
                        "Translate the Vietnamese text provided into English and provide only the translated content.\nSentence in Vietnamese:\n{}\nTranslation in English:\n",
                        "Given the sentence below, perform machine translation from Vietnamese to English. Output the translated content only.\n\nSentence in Vietnamese:\n{}\n\nTranslation in English:\n",
                        "Please translate the sentence: \"{}\" from Vietnamese to English. Output the translated content only.\n\nTranslation in English:\n",
                    ],
                    
                    'flores_zho2eng': [
                        "Translate the following sentence from Chinese to English.\n\nSentence in Chinese:\n{}\n\nTranslation in English:\n",
                        "Please translate the provided Chinese text into English. Output the translated content only.\nSentence in Chinese:\n{}\nTranslation in English:\n",
                        "Translate the Chinese text provided into English and provide only the translated content.\nSentence in Chinese:\n{}\nTranslation in English:\n",
                        "Given the sentence below, perform machine translation from Chinese to English. Output the translated content only.\n\nSentence in Chinese:\n{}\n\nTranslation in English:\n",
                        "Please translate the sentence: \"{}\" from Chinese to English. Output the translated content only.\n\nTranslation in English:\n",
                    ],
                    
                    'flores_zsm2eng': [
                        "Translate the following sentence from Malay to English.\n\nSentence in Malay:\n{}\n\nTranslation in English:\n",
                        "Please translate the provided Malay text into English. Output the translated content only.\nSentence in Malay:\n{}\nTranslation in English:\n",
                        "Translate the Malay text provided into English and provide only the translated content.\nSentence in Malay:\n{}\nTranslation in English:\n",
                        "Given the sentence below, perform machine translation from Malay to English. Output the translated content only.\n\nSentence in Malay:\n{}\n\nTranslation in English:\n",
                        "Please translate the sentence: \"{}\" from Malay to English. Output the translated content only.\n\nTranslation in English:\n",
                    ],

                    'mmlu': [
                        "Read the provided content (if exists) and respond to the question by selecting the most probable answer.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Based on the content (if provided), please directly choose the correct answer for the multiple-choice question.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Respond to the multiple-choice question with the correct answer based on the provided content.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Answer the following multi-choices question by selecting the correct option.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "As an expert, your task is to solve the following multiple-choice question by selecting the correct response from the options provided.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],

                    'c_eval': [
                        "请仔细阅读以下问题并从选项中选择最合适的答案，仅回答相应选项。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "请仔细阅读以下问题，并直接给出正确答案的选项。\n问题:\n{}\n选项:\n{}\n答案:\n",
                        "针对以下问题选择正确答案，请直接选择正确的选项。\n问题:\n{}\n选项:\n{}\n答案:\n",
                        "分析并从以下提供的选项中选择唯一的正确答案，如不确定，则选择你认为最可能的答案。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "请认真阅读以下中文多选题，并给出正确答案。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                    ],
                    
                    'cmmlu': [
                        "请仔细阅读以下问题并从选项中选择最合适的答案，仅回答相应选项。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "请仔细阅读以下问题，并直接给出正确答案的选项。\n问题:\n{}\n选项:\n{}\n答案:\n",
                        "针对以下问题选择正确答案，请直接选择正确的选项。\n问题:\n{}\n选项:\n{}\n答案:\n",
                        "分析并从以下提供的选项中选择唯一的正确答案，如不确定，则选择你认为最可能的答案。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "请认真阅读以下中文多选题，并给出正确答案。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                    ],
                    
                    'zbench': [
                        "请仔细阅读以下问题并从选项中选择最合适的答案，仅回答相应选项。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "请仔细阅读以下问题，并直接给出正确答案的选项。\n问题:\n{}\n选项:\n{}\n答案:\n",
                        "针对以下问题选择正确答案，请直接选择正确的选项。\n问题:\n{}\n选项:\n{}\n答案:\n",
                        "分析并从以下提供的选项中选择唯一的正确答案，如不确定，则选择你认为最可能的答案。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "请认真阅读以下中文多选题，并给出正确答案。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                    ],
                    
                    'ind_emotion': [
                        "Bacalah kalimat berikut dan tentukan emosinya. Pilihlah emosi yang tepat dari pilihannya.\n\nKalimat:\n{}\n\nPilihan:\n{}\n\nJawaban:\n",
                        "Klasifikasikan emosi dari kalimat berikut. Pilih jawaban yang benar dari pilihan.\nKalimat:\n{}\nPilihan:\n{}\nJawaban:\n",
                        "Tentukan label emosi dari kalimat yang diberikan dengan memilih jawaban yang benar dari pilihan.\nKalimat:\n{}\nPilihan:\n{}\nJawaban:\n",
                        "Please read the following Indonesian sentence and decide its sentiment by choosing the most possible option.\n\nSentence:\n{}\n\nChoices:\n{}\n\Answer:\n",
                        "Please determine the sentiment of the following sentence and choose the appropriate answer.\n\nSentence:\n{}\n\nChoices:\n{}\n\Answer:\n",
                    ],

                    'ocnli': [
                        "请仔细阅读以下前提和假设，并判断其关系属于蕴含、矛盾还是中性。从选项中选择最合适的答案，仅回答相应选项。\n\n前提:\n{}\n假设:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "根据以下前提和假设，判断其关系属于“蕴含”、“矛盾”、还是“中性”。请直接选择正确的选项。\n前提:\n{}\n假设:\n{}\n选项:\n{}\n答案:\n",
                        "将以下前提和假设的关系划分为“蕴含”、“矛盾”、或者“中性”。请直接选择正确的选项。\n前提:\n{}\n假设:\n{}\n选项:\n{}\n答案:\n",
                        "根据以下前提和假设的关系，选择最适合描述其关系的答案。\n\n前提:\n{}\n假设:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "阅读以下前提和假设，分析其中的关系并选择合适的答案。\n\n前提:\n{}\n假设:\n{}\n\n选项:\n{}\n\n答案:\n",
                    ],
                    
                    'c3': [
                        "请仔细阅读以下内容或对话，并回答问题。从选项中选择最合适的答案，仅回答相应选项。\n\n内容:\n{}\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "根据以下内容回答问题，请从选项中选择正确的答案。\n内容:\n{}\n问题:\n{}\n选项:\n{}\n答案:\n",
                        "仔细阅读以下内容，并回答问题。请直接选择正确的选项。\n内容:\n{}\n问题:\n{}\n选项:\n{}\n答案:\n",
                        "根据以下内容，回答相关问题，从选项中选择最合适的答案。\n\n内容:\n{}\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                        "根据内容回答以下多选题，仅有一个正确答案。\n\n内容:\n{}\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n",
                    ],

                    'dream': [
                        "Examine the dialogue and choose the suitable response to the question.\n\nDialogue:\n{}\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Carefully review the conversation and choose the correct answer directly for the multiple-choice question.\nConversation\n{}\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Based on the conversation, respond to the question by selecting the correct option.\nDialogue:\n{}\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n",
                        "You will receive a dialogue along with a question. Begin by thoroughly reading the dialogue, and then provide your answer to the question by selecting the most suitable choices.\n\nDialogue:\n{}\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Utilizing the information presented in the dialogue, respond to the question by selecting the single correct answer.\n\nDialogue:\n{}\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],

                    'samsum': [
                        "Summarize the following dialogue.\n\nDialogue:\n{}\n\nSummary:\n",
                        "Compose a concise summary by condensing the key points from the following dialogue.\nDialogue:\n{}\nSummary:\n",
                        "Please sum up the following conversation in a few sentences.\nDialogue:\n{}\nSummary:\n",
                        "Produce a brief summary of the following conversation, focusing on conveying essential information.\n\nDialogue:\n{}\n\nSummary:\n",
                        "Offer an extremely condensed summary of the conversation presented.\n\nDialogue:\n{}\n\nSummary:\n",
                        ],

                    'dialogsum': [
                        "Summarize the following dialogue.\n\nDialogue:\n{}\n\nSummary:\n",
                        "Compose a concise summary by condensing the key points from the following dialogue.\nDialogue:\n{}\nSummary:\n",
                        "Please sum up the following conversation in a few sentences.\nDialogue:\n{}\nSummary:\n",
                        "Produce a brief summary of the following conversation, focusing on conveying essential information.\n\nDialogue:\n{}\n\nSummary:\n",
                        "Offer an extremely condensed summary of the conversation presented.\n\nDialogue:\n{}\n\nSummary:\n",
                        ],


                    'sst2': [
                        "Read the following sentence and determine its sentiment. Choose the appropriate sentiment from the options provided.\n\nSentence:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Identify the sentiment of the following sentence by selecting one label from the available choices.\nSentence:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Examine the following sentence and categorize its sentiment using one of the labels provided.\nSentence:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Determine the sentiment of the following sentence and choose the most suitable option.\n\nSentence:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Respond to the following question by choosing the most suitable option.\n\nQuestion:\nDoes the following sentence convey a positive or negative sentiment?\n\nSentence:\n{}\n\nAnswer:\n",
                    ],

                    'cola': [
                        "Assess the grammatical correctness of the following sentence and choose the appropriate answer from the provided options.\n\nSentence:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Assess the grammatical accuracy of the following sentence and choose the correct answer from the available options.\nSentence:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Evaluate the grammatical correctness of the sentence and choose the correct answer from the provided choices.\nSentence:\n{}\nChoices:\n{}\nAnswer:\n",
                        "Analyze the sentence for its grammatical correctness and choose the most suitable answer from the provided options.\n\nSentence:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Respond to the following question by choosing the most suitable option.\n\nQuestion:\nIs the following sentence grammatically correct or not?\n\nSentence:\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],


                    'qqp': [
                        "Assess the semantic similarity between the following two questions and choose the appropriate answer from the provided choices.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Do the following two questions have the same meaning? Choose the correct answer from the available choices.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Choose the correct answer from the provided choices by assessing the semantic similarity of the two sentences.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Examine the following two questions and determine if they can be considered highly similar. Choose the most appropriate option.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Respond to the following question by choosing the most suitable option.\n\nQuestion:\nDo the following two questions have the same meaning?\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],


                    'mnli': [
                        "Assess the relationship between the following two sentences and choose the correct answer from the provided choices.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Determine the relationship between the following two sentences. Choose the correct answer from the available choices.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Choose the correct answer from the provided choices by assessing the relationship between the two sentences.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Examine the two provided sentences and determine their relationship by selecting the most suitable option from the given choices.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Respond to the following question by choosing the most suitable option.\n\nQuestion:\nWhat is the relationship for the following two sentences?\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],
                    
                    'qnli': [
                        "Assess whether the question can be answered based on the paragraph and choose the correct answer from the provided choices.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Based on the paragraph, can the question be answered? Choose the correct option from the provided choices.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Determine if the question can be answered from the paragraph and select the correct answer from the provided choices.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Based on the question and paragraph, determine if the answer can be inferred from the paragraph. Select the most appropriate choice as the answer.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Select the appropriate response by examining whether the question can be answered based on the provided context.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],


                    'wnli': [
                        "Assess whether the second sentence can be inferred from the first sentence and choose the correct answer from the provided choices.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Does the second sentence entail the first sentence? Choose the correct answer from the available choices.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Choose the correct answer from the provided choices by determining if the second sentence entails the first sentence.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Recognize the entailment relationship between the following sentences and choose the most appropriate answer from the provided choices.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Respond to the following question by choosing the most suitable option.\nQuestion: Are the following two sentence entailment or not?\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],

                    'rte': [
                        "Assess whether the second sentence can be inferred from the first sentence and choose the correct answer from the provided choices.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Does the second sentence entail the first sentence? Choose the correct answer from the available choices.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Choose the correct answer from the provided choices by determining if the second sentence entails the first sentence.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Recognize the entailment relationship between the following sentences and choose the most appropriate answer from the provided choices.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Respond to the following question by choosing the most suitable option.\nQuestion: Are the following two sentence entailment or not?\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],
                    
                    'mrpc': [
                        "Assess the semantic similarity between the following two sentences and choose the correct answer from the provided choices.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Do the following two sentences have the same meaning? Choose the correct answer from the available choices.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Choose the correct answer from the provided choices by assessing the semantic similarity between the two sentences.\n{}\nChoices:\n{}\nAnswer:\n",
                        "Do the sentences have the same meaning? Select the most suitable answer.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                        "Do the following two sentences convey the same meaning? Choose the most appropriate answer.\n\n{}\n\nChoices:\n{}\n\nAnswer:\n",
                    ],

}


