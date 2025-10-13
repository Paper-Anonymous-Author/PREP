import LLM_prompt
import config
import argparse

parser = argparse.ArgumentParser()
# Dataset settings
parser.add_argument("--dataset", type=str, default='douban',
                    help="Select dataset for model")
parser.add_argument("--task", type=str, default='init',
                    help="Select task")

args = parser.parse_args()
cfg = config.Config(args=args)

dataset_now = args.dataset
dataset_type = 'douban' if dataset_now=='douban' else 'amz'
task_now = args.task

assert  task_now in ['init', 'finetune', 'generate', 'evaluate']

if task_now == 'init':
    llm_client = LLM_prompt.prepare_gpt(cfg)
    LLM_prompt.LLM_generate(cfg, llm_client, 1, True)

if task_now == 'finetune':
    llm_client = LLM_prompt.prepare_gpt(cfg)
    LLM_prompt.LLM_generate_no_batch(cfg, llm_client, True)

if task_now == 'generate':
    llm_client = LLM_prompt.prepare_gpt(cfg)
    LLM_prompt.evaluate_with_LLM(cfg, llm_client, 'generate', dataset_type)
if task_now == 'evaluate':
    llm_client = LLM_prompt.prepare_gpt(cfg)
    LLM_prompt.evaluate_with_LLM(cfg, llm_client, 'evaluate', dataset_type)
