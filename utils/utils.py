import json
import os
import random
import numpy as np
import torch
import time
# import anthropic
# import google.generativeai as genai
import openai
from openai import OpenAI



# API setting constants
API_MAX_RETRY = 25
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

def chat_completion_openai(model, messages, temperature, max_tokens):
    client = OpenAI()
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # remove system prompt for o1 models, use allowable sampling parameters
            if "o1" in model or "o3-" in model:
                if messages[0]['role'] == 'system':
                    messages = messages[1:]
                response = client.chat.completions.create(
                    model=model, messages=messages, n=1, temperature=1
                )

            else:
                response = client.chat.completions.create(
                    model=model, messages=messages, n=1, temperature=temperature, max_tokens=max_tokens
                )

            output = response.choices[0].message.content
            break
        except openai.APIError as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            time.sleep(API_RETRY_SLEEP)

        except openai.APIConnectionError as e:
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            time.sleep(API_RETRY_SLEEP)

        except openai.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            time.sleep(API_RETRY_SLEEP)

    return output



def read_jsonl(file_path):
    with open(file_path, 'r') as fr:
        data = [json.loads(line) for line in fr.readlines()]
    print(len(data))
    print(data[0].keys())
    return data

def write_jsonl(data, jfile, skip_none=False):
    with open(jfile, 'w', encoding='utf-8') as f:
        for d in data:
            if d is None and not skip_none:
                raise ValueError('None object !')
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f'Wrote {len(data)} -> {jfile}')
    

def get_ragas_pairwise(results_pos, results_neg, split='factuality'):
    '''
    From RAGAS metrics, compute pairwise comparison output using evaluation hierarchy based on splits
    If factuality split:
      Response A more factual than Response B --> A wins
      Else, wrong
    If completeness split:
      Response A and Response B equally factual AND Response A more relevant
      Else, wrong
    We only do this for factuality and completeness splits
    '''
    judgements_1, judgements_2 = [], []
    output_scores = []

    for rp, rn in zip(results_pos, results_neg):
        rp_faithful, rp_rel = rp['faithfulness'], rp['answer_relevancy']
        rn_faithful, rn_rel = rn['faithfulness'], rn['answer_relevancy']
        rp_faithful = rp_faithful if rp_faithful is not None else 0.0
        rn_faithful = rn_faithful if rn_faithful is not None else 0.0
        rp_rel = rp_rel if rp_rel is not None else 0.0
        rn_rel = rn_rel if rn_rel is not None else 0.0

        # Score output for debug / analysis
        score_text = f'positive_faithful: {rp_faithful} | positive_relevancy: {rp_rel} | negative_faithful: {rn_faithful} | negative_relevancy: {rn_rel}'
        output_scores.append(score_text)

        #TODO: change split name to 'factuality' or whatever we name the faithfulness script
        if 'faithfulness' in split or 'refusal' in split: 
            # if positive response more faith, correct
            if rp_faithful > rn_faithful:
                judgements_1.append(1)
                judgements_2.append(1)
            else:
                judgements_1.append(2)
                judgements_2.append(2)
        
        elif 'completeness' in split:
            if (rp_faithful == rn_faithful and
                rp_rel > rn_rel):
                    judgements_1.append(1)
                    judgements_2.append(1)

            else:
                judgements_1.append(2)
                judgements_2.append(2)

    return judgements_1, judgements_2, output_scores


def compute_acc(judgements_1, judgements_2):
    '''
    Computes consistent accuracy
    judgments_{1,2}: list of length N samples
    - Each element of judgements_i can be a list or an int
    - list: Accuracy computed based on self-consistency
    - int: Regular accuracy computation.
    '''
    # Compute accuracy metrics
    accuracy_consistent = []
    accuracy_swap1 = []
    accuracy_swap2 = []
    consistency = []
    for j1, j2 in zip(judgements_1, judgements_2):

        # consistent
        if isinstance(j1, list) and isinstance(j2, list):
            if len(j1) == 1 and len(j2) == 1:
                j1 = j1[0]
                j2 = j2[0]

        if not isinstance(j1, list) and not isinstance(j2, list):
            if j1 == j2:
                consistency.append(1)

                # Label for all pairs is 1
                if j1 == 1:
                    accuracy_consistent.append(1)
                else:
                    accuracy_consistent.append(0)

            # Not consistent, no accuracy credit
            else:
                consistency.append(0)
                accuracy_consistent.append(0)

            # Accuracy of the first run
            if j1 == 1:
                accuracy_swap1.append(1)
            else:
                accuracy_swap1.append(0)

            # Accuracy of the second run
            if j2 == 1:
                accuracy_swap2.append(1)
            else:
                accuracy_swap2.append(0)
                
        else:
            assert isinstance(j1, list) and isinstance(j2, list)
            # self-consistency, j1, j2 are lists

            consistency_n = []
            accuracy_consistent_n = []
            accuracy_swap1_n = []
            accuracy_swap2_n = []
            for k1, k2 in zip(j1, j2):
                if k1==k2:
                    consistency_n.append(1)

                    # Label for all pairs is 1
                    if k1 == 1:
                        accuracy_consistent_n.append(1)
                    else:
                        accuracy_consistent_n.append(0)

                else:
                    consistency_n.append(0)
                    accuracy_consistent_n.append(0)

                if k1 == 1:
                    accuracy_swap1_n.append(1)
                else:
                    accuracy_swap1_n.append(0)

                # Accuracy of the second run
                if k2 == 1:
                    accuracy_swap2_n.append(1)
                else:
                    accuracy_swap2_n.append(0)
            
            consistency.append(consistency_n)
            accuracy_consistent.append(accuracy_consistent_n)
            accuracy_swap1.append(accuracy_swap1_n)
            accuracy_swap2.append(accuracy_swap2_n)


    return accuracy_consistent, consistency, accuracy_swap1, accuracy_swap2