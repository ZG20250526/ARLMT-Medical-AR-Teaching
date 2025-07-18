# import sys
# import json
# import argparse
# from pprint import pprint
# from copy import deepcopy
# from collections import defaultdict
#
# sys.path.append("llava")
# from llava.openai_api import call_async
#
#
# class LLMEvalPromptGenerator:
#
#   instruct_prompt = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with caption describing the same image.
#     Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
#     Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""
#   role = 'Assistant'
#
#   @staticmethod
#   def conv_to_str(fig_label, fig_caption, fig_inline_mention, question, ans1, ans2):
#     return (f'[Context]\n'
#             f'Figure Caption:\n{fig_label}: {fig_caption}\n\n'
#             f'Figure Context:\n\t- {fig_inline_mention}\n\n'
#             f'[Question]\n{question}\n\n'
#             f'[{LLMEvalPromptGenerator.role} 1]\n{ans1}\n\n[End of {LLMEvalPromptGenerator.role} 1]\n\n'
#             f'[{LLMEvalPromptGenerator.role} 2]\n{ans2}\n\n[End of {LLMEvalPromptGenerator.role} 2]\n\n'
#             f'[System]\n{LLMEvalPromptGenerator.instruct_prompt}\n\n')
#
#   @staticmethod
#   def compare_messages_gen(sample):
#     messages = [
#     {"role": "system", "content": """'You are a helpful and precise assistant for checking the quality of the answer."""},
#     ]
#     messages.append({"role": "user", "content": LLMEvalPromptGenerator.conv_to_str(sample['fig_label'], sample['fig_caption'], sample['in_text_mention'], sample['question'], sample['ans1'], sample['ans2'])})
#     return messages
#
#
# class ChatEvaluation:
#   # Calculate precision, recall, F1 overall and for each domain.
#
#   @staticmethod
#   def get_domain(x):
#     for domain in ['chest_xray', 'mri', 'histology', 'gross', 'ct_scan']:
#       in_domain = x['domain'][domain]
#       if in_domain:
#         return domain
#
#   @staticmethod
#   def get_avg(x):
#     return sum([float(y) for y in x])/len(x)
#
#   @staticmethod
#   def eval(samples):
#     predictions = [(x['question_id'], x['type'], ChatEvaluation.get_domain(x), x['result'].split('\n')[0].split(' ')) for x in samples]
#     score_type_dict = defaultdict(lambda: defaultdict(list))
#     for q_id, q_type, domain, (a1_score, a2_score) in predictions:
#       score_type_dict[q_type][1].append(a1_score)
#       score_type_dict[q_type][2].append(a2_score)
#       score_type_dict['all'][1].append(a1_score)
#       score_type_dict['all'][2].append(a2_score)
#       score_type_dict[domain][1].append(a1_score)
#       score_type_dict[domain][2].append(a2_score)
#
#     result = defaultdict(dict)
#
#     for q_type, score_dict in score_type_dict.items():
#       result[q_type]['gpt4_score'] = ChatEvaluation.get_avg(score_dict[1])
#       result[q_type]['pred_score'] = ChatEvaluation.get_avg(score_dict[2])
#       result[q_type]['pred_relative_score'] = ChatEvaluation.get_avg([float(s2)/float(s1) for s1, s2 in zip(score_dict[1], score_dict[2])])*100
#       result[q_type]['data_size'] = len(score_dict[1])
#     # print results
#     pprint(result)
#
#
# def main(args):
#   # Load input data
#   answer_data = []
#   with open(args.input_path) as f:
#     for line in f:
#       answer_data.append(json.loads(line))
#
#   question_data = []
#   with open(args.question_input_path) as f:
#     for line in f:
#       question_data.append(json.loads(line))
#
#   # Merge question and answer input data
#   samples = []
#   for question, answer in zip(question_data, answer_data):
#     sample = deepcopy(question)
#     question['question'] = sample['text'][:-8]
#     question['ans1'] = sample.pop('gpt4_answer')
#     question['ans2'] = answer['text']
#     samples.append(question)
#
#   samples_question_ids = set(x['question_id'] for x in samples)
#
#   # Generate GPT-4 evaluation of indivdual answers between model answer and GPT-4 answer
#   results = []
#   BATCH_SIZE = 3
#   for i in range(30):
#     result_question_ids = set(result['question_id'] for result in results)
#
#     batch = []
#     counter = 0
#     for sample in samples:
#       if sample['question_id'] in result_question_ids:
#         continue
#       batch.append(sample)
#       if len(batch)>=BATCH_SIZE:
#         async_results = call_async(batch, lambda x: LLMEvalPromptGenerator.compare_messages_gen(x))
#         results.extend(async_results)
#         print(f"Result Size: {len(results)}")
#         batch = []
#     async_results = call_async(batch, lambda x: LLMEvalPromptGenerator.compare_messages_gen(x))
#     results.extend(async_results)
#     print(f"Result Size: {len(results)}")
#
#   # Print number of questions and results
#   print(f'all samples: {len(samples_question_ids)}')
#   print(f'ran samples: {len(result_question_ids)}')
#   print(f'to be run samples: {len(samples_question_ids-result_question_ids)}')
#
#   # Write GPT-4 evaluation outputs to output_path
#   with open(args.output_path, 'w') as f:
#     for line in results:
#       f.write(json.dumps(line)+'\n')
#
#   # Perform Evaluation for all results
#   ChatEvaluation().eval(results)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--question_input_path', type=str, default='data/eval/llava_med_eval_qa50_qa.jsonl')
#     parser.add_argument('--input_path', type=str, default='dbfs:/mnt/hanoverdev/scratch/clwon/llava/test/answers/test50/2023-05-10_med-pretrain-364m-v1-1epoch.jsonl')
#     parser.add_argument('--output_path', type=str, default='data/eval/llava_med_eval_qa50_qa_ans.jsonl')
#     args = parser.parse_args()
#     main(args)
import os
import json
import argparse
from copy import deepcopy
import itertools
from typing import Any
from operator import add
from pprint import pprint
from typing import List
from pathlib import Path
from tqdm import tqdm

import llm
import util


INSTRUCT_PROMPT = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with caption describing the same image.
  Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
  Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""
ROLE = 'Assistant'

# Generate instruction for GPT-4 to score the two answers.
def conv_to_str(fig_label, fig_caption, fig_context, question, ans1, ans2):
  return (f'[Context]\n'
          f'Figure Caption:\n{fig_label}: {fig_caption}\n\n'
          f'Figure Context:\n\t- {fig_context}\n\n'
          f'[Question]\n{question}\n\n'
          f'[{ROLE} 1]\n{ans1}\n\n[End of {ROLE} 1]\n\n'
          f'[{ROLE} 2]\n{ans2}\n\n[End of {ROLE} 2]\n\n'
          f'[System]\n{INSTRUCT_PROMPT}\n\n')

def compare_messages_gen(fig_label, fig_caption, fig_context, question, ans1, ans2):
  messages = [
  {"role": "system", "content": """'You are a helpful and precise assistant for checking the quality of the answer."""},
  ]
  messages.append({"role": "user", "content": conv_to_str(fig_label, fig_caption, fig_context, question, ans1, ans2)})
  return messages


def sum_list_list(x):
  return sum(item for inner_list in x for item in inner_list)

def chunk(lst, n):
  for i in range(0, len(lst), n):
    if i+(1.5*n)<len(lst):
      end = i + n
    else:
      end = len(lst)
    yield lst[i:end]
    if end==len(lst):
      return


def infer(samples):
    model_inst = llm.GPT("gpt-4-0314")

    BATCH_SIZE = 1
    batch_samples = []
    results = []
    batch = []

    print('Starting Multimodal Chat GPT Scoring Eval')

    for sample in tqdm(samples):
        sample_copy = deepcopy(sample)
        input_msg = compare_messages_gen(sample_copy['fig_label'], sample_copy['fig_caption'], sample_copy['in_text_mention'], sample_copy['question'], sample_copy['ans1'], sample_copy['ans2'])
        batch.append(input_msg)
        batch_samples.append(sample_copy)
        if len(batch)>=BATCH_SIZE:
            inference_results = [x.strip() for chunk_messages in chunk([x for x in batch if x], BATCH_SIZE) for x in model_inst.infer(chunk_messages)]
            for item, inference_result in zip(batch_samples, inference_results):
                item['gpt_eval'] = inference_result
            results.extend(batch_samples)
            batch = []
            batch_samples = []
    inference_results = [x.strip() for chunk_messages in chunk([x for x in batch if x], BATCH_SIZE) for x in model_inst.infer(chunk_messages)]
    for item, inference_result in zip(batch_samples, inference_results):
        item['gpt_eval'] = inference_result
    results.extend(batch_samples)
    print(f"Result Size: {len(results)}")
    return results


def main(args):
    answer_data = util.load_file_jsonl(args.answers_file)
    question_data = util.load_file_jsonl(args.question_file)

    samples = []
    for question, answer in zip(question_data, answer_data):
        question_copy = deepcopy(question)
        question['question'] = question_copy['text']
        question['ans1'] = question_copy.pop('gpt4_answer')
        question['ans2'] = answer['text']
        samples.append(question)

    results = infer(samples)

    # Create parent directory of output score files if it doesn't exist
    os.makedirs(Path(args.scores_file).parent, exist_ok=True)

    with open(args.scores_file, 'w') as f:
       for row in results:
          f.write(json.dumps(row)+'\n')


if __name__ == '__main__':
   parser = argparse.ArgumentParser("GPT-4 Multimodal Chat Scoring", add_help=True)
   parser.add_argument("--answers-file", default="", metavar="FILE", help="path to model answer file")
   parser.add_argument("--question-file", default="data/questions/llava_med_eval_qa50_qa.jsonl", metavar="FILE", help="path to multichat questions file")
   parser.add_argument("--scores-file", default="", metavar="FILE", help="path to save gpt-4 score file")
   args = parser.parse_args()
   main(args)