import torch
import json
import os
import argparse
import tqdm
import numpy as np
import importlib


# Adopted from RewardBench: https://github.com/allenai/reward-bench/blob/main/rewardbench/generative.py
ANTHROPIC_MODEL_LIST = (
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-instant-1",
    "claude-instant-1.2",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620",
)

OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "gpt-4o",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o1",
    "o1-mini",
)

GEMINI_MODEL_LIST = (
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
    "gemini-1.5-pro-exp-0801",
    "gemini-1.5-pro-exp-0827",
    "gemini-1.5-flash-exp-0827",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-8b-exp-0827",
)

API_MODEL_LIST = OPENAI_MODEL_LIST + ANTHROPIC_MODEL_LIST + GEMINI_MODEL_LIST



###
# VLLM PROMPTED JUDGE UTILS BELOW
###

DEFAULT_CRITERIA="completeness. A response is complete if it addresses all aspects of the question.\nIf two responses are equally complete, evaluate based on conciseness. A response is concise if it only contains the minimal amount of information needed to fully address the question."
CONCISENESS_CRITERIA="conciseness. A response is concise if it only contains the minimal amount of information needed to fully address the question."
COMPLETENESS_CRITERIA="completeness. A response is complete if it addresses all aspects of the question."
REFUSAL_CRITERIA="refusal validity. A response correctly refuses to answer if the context does not contain sufficient information to answer the user's question."
FAITHFULNESS_CRITERIA="faithfulness.  A response is faithful to the context if all of the factual information in the response is attributable to the context."

criteria_dict = {
    'default': DEFAULT_CRITERIA,
    'conciseness': CONCISENESS_CRITERIA,
    'completeness': COMPLETENESS_CRITERIA,
    'refusal': REFUSAL_CRITERIA,
    'faithfulness': FAITHFULNESS_CRITERIA,
}


model_to_file = {
    'Skywork/Skywork-Critic-Llama-3.1-8B': 'skywork',
    'Skywork/Skywork-Critic-Llama-3.1-70B': 'skywork',
    'facebook/Self-taught-evaluator-llama3.1-70B/dpo_model': 'selftaught',
    'NCSOFT/Llama-3-OffsetBias-8B': 'offsetbias',
    'prometheus-eval/prometheus-7b-v2.0': 'prometheus',
    'prometheus-eval/prometheus-8x7b-v2.0': 'prometheus',
    'Skywork/Skywork-Reward-Llama-3.1-8B': 'skywork_rm',
    'PatronusAI/glider': 'glider',
    'AtlaAI/Selene-1-Mini-Llama-3.1-8B': 'atla',
}

def load_prompts_and_parsing(model, prompt_strategy):
    prompt_template = ''
    get_judgement = None

    if model not in model_to_file and model not in API_MODEL_LIST:
        model_to_file[model] = 'default_prompts' #default prompt is sfr judge
        print('Model not added explicitly; using default prompts (same as SFRJudge/Atla prompts)!')

    if model in model_to_file:
        filename = model_to_file[model]

        try: 
            if prompt_strategy == 'vanilla':
                if filename == 'selftaught':
                    prompt_system = importlib.import_module('utils.prompts.{}'.format(filename)).PROMPT_PAIRWISE_SYSTEM
                    prompt_input = importlib.import_module('utils.prompts.{}'.format(filename)).PROMPT_PAIRWISE
                    prompt_template = f"{prompt_system}<PROMPT_SYSTEM_SPLIT_STRING>{prompt_input}"
                else:
                    prompt_template = importlib.import_module('utils.prompts.{}'.format(filename)).PROMPT_PAIRWISE
                get_judgement = importlib.import_module('utils.prompts.{}'.format(filename)).pairwise_parsing
            elif prompt_strategy == 'conditional':
                if filename == 'selftaught':
                    prompt_system = importlib.import_module('utils.prompts.{}'.format(filename)).PROMPT_PAIRWISE_SYSTEM_CONDITIONAL
                    prompt_input = importlib.import_module('utils.prompts.{}'.format(filename)).PROMPT_PAIRWISE
                    prompt_template = f"{prompt_system}<PROMPT_SYSTEM_SPLIT_STRING>{prompt_input}"
                else:
                    prompt_template = importlib.import_module('utils.prompts.{}'.format(filename)).PROMPT_PAIRWISE_CONDITIONAL
                get_judgement = importlib.import_module('utils.prompts.{}'.format(filename)).pairwise_parsing_conditional
            elif prompt_strategy == 'criteria_specific':
                if filename == 'selftaught':
                    prompt_system = importlib.import_module('utils.prompts.{}'.format(filename)).PROMPT_PAIRWISE_SYSTEM_GENERIC
                    prompt_input = importlib.import_module('utils.prompts.{}'.format(filename)).PROMPT_PAIRWISE
                    prompt_template = f"{prompt_system}<PROMPT_SYSTEM_SPLIT_STRING>{prompt_input}"
                else:
                    prompt_template = importlib.import_module('utils.prompts.{}'.format(filename)).PROMPT_PAIRWISE_GENERIC
                get_judgement = importlib.import_module('utils.prompts.{}'.format(filename)).pairwise_parsing
        except Exception as e:
            print(f'Failed to load template with error: {e}')

        return prompt_template, get_judgement

    # API model: load default prompt
    else:
        from utils.prompts.default_prompts import PROMPT_PAIRWISE as prompt_template
        from utils.prompts.default_prompts import pairwise_parsing as get_judgement

        return prompt_template, get_judgement