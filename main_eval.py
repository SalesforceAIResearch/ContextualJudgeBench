import json,os,time,argparse,importlib,glob,string,gc,torch,sys
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, load_dataset

from transformers import AutoTokenizer
from vllm import SamplingParams

from utils.prompt_utils import load_prompts_and_parsing, OPENAI_MODEL_LIST, API_MODEL_LIST
from utils.utils import chat_completion_openai, compute_acc
from model import VllmEndpoint


HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def main(args):
    ### Model loading ---------------------------------------------------------------------
    # Check if judge is queried via API 
    is_api_models = isinstance(args.judge_model, list) or (args.judge_model in API_MODEL_LIST)

    # Load judge prompt
    judge_prompt, get_judgement = load_prompts_and_parsing(args.judge_model, args.prompt_strategy)
    if judge_prompt == '':
        raise Exception('No judge prompt loaded!')
        
    # Self-taught-evaluator specifically utilizes a system prompt
    judge_system_prompt = ''
    if '<PROMPT_SYSTEM_SPLIT_STRING>' in judge_prompt:
        judge_system_prompt, judge_prompt = judge_prompt.split('<PROMPT_SYSTEM_SPLIT_STRING>')

    ### Model loading done -----------------------------------------------------------------


    # Load eval dataset --------------------------------------------------------------------
    # features: ['negative_response', 'positive_response', 'context', 'question', 'question_id'],
    # question_id unused, just for logging source details
    eval_dataset_full = load_dataset("Salesforce/ContextualJudgeBench")
    VALID_DATASET_SPLITS = [
        'completeness_qa', 'completeness_summ', 
        'conciseness_qa', 'conciseness_summ', 
        'faithfulness_qa', 'faithfulness_summ', 
        'refusal_answerable', 'refusal_unanswerable'
        ]
        

    if "all" in args.splits:
        args.splits = VALID_DATASET_SPLITS

    assert all(s in VALID_DATASET_SPLITS for s in args.splits), f"Found an invalid split! Please ensure your input list only contains `all` or one of {VALID_DATASET_SPLITS}"

    # Iterate over splits, run inference!
    for split in args.splits:
        eval_dataset = eval_dataset_full[split]
            
        if args.debug:
            eval_dataset = eval_dataset.select(range(10))

        ### Make output dir ---------------------------------------------------------------------
        if not os.path.isdir(os.path.join(args.output_path, args.prompt_strategy)):
            print(f"Creating directory: {args.output_path}")
            os.makedirs(os.path.join(args.output_path, args.prompt_strategy),exist_ok=True)

        output_file = os.path.join(args.output_path, args.prompt_strategy, f'eval_detailed_{split}.jsonl')
        if args.num_sequences > 1:
            output_file = output_file.replace('.jsonl', f'.n{args.num_sequences}.jsonl')
        if  os.path.exists(output_file):
            print(f"File already exists for split {split}; skipping!")
            continue

        # Set the evaluation criteria based on prompt strategy and split
        evaluation_criteria = ''
        from utils.prompt_utils import criteria_dict
        if args.prompt_strategy in ['vanilla', 'conditional']:
            evaluation_criteria = criteria_dict['default']
        elif args.prompt_strategy == 'criteria_specific':
            candidate_splits = ['conciseness', 'completeness', 'refusal', 'faithfulness']
            assert any(cs in split for cs in candidate_splits), f'Split not in {candidate_splits}'
            
            for cs in candidate_splits:
                if cs in split:
                    evaluation_criteria = criteria_dict[cs]
                    break

        assert evaluation_criteria != '', 'Evaluation criteria empty! Invalid split and prompt strategy combination'

        print(f"Using evaluation critera: {evaluation_criteria}")
        # Inference!
        if is_api_models:
            # Implementation adopted from RewardBench: https://github.com/allenai/reward-bench/blob/main/scripts/run_generative.py
            ####################################
            # Run judge model via API access
            ####################################
            if args.judge_model in OPENAI_MODEL_LIST:
                chat_completion = chat_completion_openai
            
            def update_progress_bar(done, total):
                # Simple text-based progress bar
                progress = int(50 * done / total)  # Calculate progress (50 chars width)
                sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
                sys.stdout.flush()

            def get_api_output(example, idx):
                output_keys = [('positive_response', 'negative_response'), ('negative_response', 'positive_response')]
                pair = output_keys[idx]

                content = judge_prompt.format(
                    criteria=evaluation_criteria,
                    question=example['question'],
                    response_a=example[pair[0]],
                    response_b=example[pair[1]],
                    context=example['context'],
                )

                if judge_system_prompt == '':
                    messages = [
                        {"role": "user", "content": content}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": judge_system_prompt},
                        {"role": "user", "content": content}
                    ]
                
                # query api
                output = chat_completion(args.judge_model, messages, args.temperature, 1024)

                # return output
                return output

            pair_inferences = []

            # Run each inference twice, swapping response order
            for idx in [0, 1]:
                with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                    inferences_idx = [None] * len(eval_dataset) # Preallocate output list
                    done_tasks = 0

                    future_to_index = {executor.submit(get_api_output, x, idx): i for i, x in enumerate(eval_dataset)}

                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        inferences_idx[index] = future.result()
                        done_tasks += 1
                        update_progress_bar(done_tasks, len(eval_dataset))

                    pair_inferences.append(inferences_idx)
                print()
            
            # Parse out the judgement from each output using get_prediction loaded from prompt_utils
            # By construction, votes_1 is inference when positive_response is position one, so correct label for all judgements is 1
            judgements_1 = [get_judgement(inf, flip=False) for inf in pair_inferences[0]] 
            judgements_2 = [get_judgement(inf, flip=True) for inf in pair_inferences[1]]
            outputs_1, outputs_2 = pair_inferences[0], pair_inferences[1]


        # Regular prompted judge inference
        elif args.judge_model not in ['ragas', 'minicheck']:
            ####################################
            # Run judge model locally with VLLM!
            ####################################

            # Load model
            os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
            if 'mistralai' in args.judge_model or 'prometheus' in args.judge_model.lower():
                os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS' # needed for sliding window (mistral/prometheus)


            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
            # if "llama-3" in args.judge_model.lower() or "llama3" in args.judge_model.lower():
            #     stop_token_ids = [128009]
            # else:
            #     stop_token_ids = []

            print('-'*50)
            print(f'Evaluating: [{args.judge_model}] for split [{split}] with prompt [{args.prompt_strategy}]')
            print('-'*50)
            max_tokens=2048 if "DeepSeek-R1" in args.judge_model else 1024
            sampling_params = SamplingParams(
                n=args.num_sequences,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_tokens,
            )
            
            # back compatible port argument
            if args.port is not None:
                base_url = f"http://localhost:{port}/v1"
                args.base_url = base_url
                print("WARNING: Argument `port` specified. Overriding any specified `base_url` argument and using `http:localhost:\{args.port\}/v1 as endpoint")

            model = VllmEndpoint(
                args.judge_model, 
                args.base_url,
                sampling_params,
                args.api_key,
            )

            print(f"Loaded model!")
            # Apply chat template for all samples
            def process_example(example):

                output_keys = [('positive_response', 'negative_response'), ('negative_response', 'positive_response')]
                swap_outputs, swap_judgements = [], []
                for idx, pair in enumerate(output_keys):
                    
                    if judge_system_prompt == '':
                        content = judge_prompt.format(
                            criteria=evaluation_criteria,
                            question=example['question'],
                            response_a=example[pair[0]],
                            response_b=example[pair[1]],
                            context=example['context'],
                        )
                        messages = [
                            {"role": "user", "content": content}
                        ]
                    else:
                        content_system = judge_system_prompt
                    
                        content = judge_prompt.format(
                            criteria=evaluation_criteria,
                            question=example['question'],
                            response_a=example[pair[0]],
                            response_b=example[pair[1]],
                            context=example['context'],
                        )
                        messages = [
                            {"role": "system", "content": judge_system_prompt},
                            {"role": "user", "content": content}
                        ]

        
                    input_seq = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inferences = model.generate(messages)
                    
                    flip = False if idx == 0 else True
                    judgement = [get_judgement(inf, flip=flip) for inf in inferences]
                
                    swap_outputs.append(inferences)
                    swap_judgements.append(judgement)

                output = {
                    'judgement_1': swap_judgements[0],
                    'judgement_2': swap_judgements[1],
                    'output_1': swap_outputs[0],
                    'output_2': swap_outputs[1],
                }

                return output

            print("Beginning model inference!")
            updated_dataset = eval_dataset.map(process_example, num_proc=10)

            print("Completed model inference!")
            judgements_1, judgements_2 = updated_dataset['judgement_1'], updated_dataset['judgement_2']
            outputs_1, outputs_2 = updated_dataset['output_1'], updated_dataset['output_2']
        
        elif args.judge_model == 'ragas':
            #raise Exception('RAGAS not implemented yet')
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas import evaluate, EvaluationDataset
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.metrics import Faithfulness, ResponseRelevancy
            from utils.utils import get_ragas_pairwise

            evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
            evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

            def format_example(example, response="positive"):
                response = example[f"{response}_response"]
                context = example["context"]
                question = example["question"]

                example["user_input"] = question
                example["response"] = response
                example["retrieved_contexts"] = [context]

                return example

            # RAGAS operates pointwise, so we run evaluation for both positive and negative responses pointwise
            formatted_dataset_pos = eval_dataset.map(format_example, num_proc=10, fn_kwargs={"response": "positive"})
            formatted_dataset_pos = formatted_dataset_pos.select_columns(["user_input", "response", "retrieved_contexts"])
            evaluation_dataset_pos = EvaluationDataset.from_hf_dataset(formatted_dataset_pos)
            result_pos = evaluate(
                dataset=evaluation_dataset_pos,
                metrics=[Faithfulness(), ResponseRelevancy()],
                llm=evaluator_llm,
                embeddings=evaluator_emb
            ).to_pandas()
            result_pos = Dataset.from_pandas(result_pos)

            formatted_dataset_neg = eval_dataset.map(format_example, num_proc=10, fn_kwargs={"response": "negative"})
            formatted_dataset_neg = formatted_dataset_neg.select_columns(["user_input", "response", "retrieved_contexts"])
            evaluation_dataset_neg = EvaluationDataset.from_hf_dataset(formatted_dataset_neg)
            result_neg = evaluate(
                dataset=evaluation_dataset_neg,
                metrics=[Faithfulness(), ResponseRelevancy()],
                llm=evaluator_llm,
                embeddings=evaluator_emb
            ).to_pandas()
            result_neg = Dataset.from_pandas(result_neg)

            # We convert RAGAS metrics into pairwise scores using our hierarchy
            judgements_1, judgements_2, outputs_1 = get_ragas_pairwise(result_pos, result_neg, split=split)
            outputs_2 = ['' for _ in range(len(judgements_2))]

        elif args.judge_model == 'minicheck':
            from minicheck.minicheck import MiniCheck
            scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', enable_prefix_caching=False)
        
            judgements_1, outputs_1 = [], []

            for example in tqdm(eval_dataset):
                pred_label, raw_prob, _, _ = scorer.score(
                    docs=[f"{example['question']} \n\n {example['context']}", f"{example['question']} \n\n {example['context']}"], 
                    claims=[example['positive_response'], example['negative_response']])
                outputs_1.append((pred_label, raw_prob))            
                judgements_1.append(int(raw_prob[0] > raw_prob[1])) # Based on predicted probs

            # minicheck does pointwise grading, so just repeat judgements and outputs for "second consistency run"
            outputs_2 = outputs_1
            judgements_2 = judgements_1


        # Compute consistent accuracy
        #   accuracy_swap{1,2} are consistency Run 1 and Run 2 accuracies, respectively.
        accuracy_consistent, consistency, accuracy_swap1, accuracy_swap2 = compute_acc(judgements_1, judgements_2)
        output_path = os.path.join(args.output_path, args.prompt_strategy, f'eval_detailed_{split}.jsonl')

        if args.debug:
            output_path = output_path.replace('.jsonl', '.debug.jsonl')
        if args.num_sequences > 1:
            output_path = output_path.replace('.jsonl', f'.n{args.num_sequences}.jsonl')

        with open(output_path, 'w', buffering=1) as fw:
            for i in range(len(eval_dataset)):
                output = {
                    "votes": [judgements_1[i], judgements_2[i]],
                    "swap_inference1": outputs_1[i],
                    "swap_inference2": outputs_2[i],
                }
            
                fw.write(json.dumps(output)+"\n")

        if args.num_sequences == 1:
            output_evaluation = {
                'accuracy': float(np.average(accuracy_consistent)),
                'consistency': float(np.average(consistency)),
                'swap1_accuracy': float(np.average(accuracy_swap1)),
                'swap2_accuracy': float(np.average(accuracy_swap2)),
            }

            print("Completed output evaluation!")
            print(output_evaluation)

            output_result_path = os.path.join(args.output_path, args.prompt_strategy, f'eval_detailed_{split}.json')
            with open(output_result_path, "w") as fw:
                json.dump(output_evaluation, fw, indent=4, sort_keys=True)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Evaluation parameters
    parser.add_argument("--output_path", type=str, help="the directory for storing results")
    parser.add_argument("--splits", nargs="+", help="List of splits you want to run; to run all, set as [all]", default=["all"], required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples for openai api calls")
    parser.add_argument("--api_key", type=str, default="sample-api-key")

    # judge info
    parser.add_argument("--judge_model", type=str, help="model checkpoint or name")
    parser.add_argument("--prompt_strategy", type=str, default='vanilla', choices=['vanilla', 'conditional', 'criteria_specific'], help='type of prompting for judge models')
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="base url for served VLLM judge")
    parser.add_argument("--port", type=int, default=None, help="If used, will override endpoint specified in base_url and use localhost:{port} for backwards compatibility!")

    # decoding strategy
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--num_sequences", default=1, type=int)

    args = parser.parse_args()

    main(args)