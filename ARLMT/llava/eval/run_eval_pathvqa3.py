import argparse
import json
import collections
import random
import pandas as pd    
from nltk.translate.bleu_score import sentence_bleu
from eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate
from eval_metrics.glossary import *

import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file', )
    parser.add_argument('--candidate', type=str, default="candidate.json", help='path to candidate answer file', )
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data

def evaluate(gt, pred, candidate, criterion=None):    
    closed_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    open_hit_scores = collections.defaultdict(list)

    for gt_item, pred_item in zip(gt, pred):
        try:
            gt_results = gt_item['conversations']
        except:
            gt_results = gt_item['conversatons']
        gt_value = gt_results[1]['value'].lower()
        pred_value = pred_item['text'].lower()

        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        # 判断是否为封闭式问题 (Yes/No)
        if gt_value in ["yes", "no"]:
            # 处理封闭式问题
            closed_scores['q_id'].append(pred_item['question_id'])
            if gt_value in pred_value:
                closed_scores['hit'].append(1)  # 预测正确
            else:
                closed_scores['hit'].append(0)  # 预测错误
        else:
            # 处理开放式问题
            open_hit_scores['hit'].append(calculate_appearance_with_normalization(pred_value, gt_value, candidate))
            open_hit_scores['q_id'].append(pred_item['question_id'])

            exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
            exact_scores['q_id'].append(pred_item['question_id'])

            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores['f1'].append(f1_score)
            f1_scores['precision'].append(precision)
            f1_scores['recall'].append(recall)
            f1_scores['q_id'].append(pred_item['question_id'])

            # 计算 BLEU 得分
            b_score = sentence_bleu([gt_value.split()], pred_value.split())
            bleu_scores['bleu_score'].append(b_score)
            bleu_scores['q_id'].append(pred_item['question_id'])

            # 计算其他 BLEU 子得分
            b_score_1 = sentence_bleu([gt_value.split()], pred_value.split(), weights=(1, 0, 0, 0))
            b_score_2 = sentence_bleu([gt_value.split()], pred_value.split(), weights=(0, 1, 0, 0))
            b_score_3 = sentence_bleu([gt_value.split()], pred_value.split(), weights=(0, 0, 1, 0))
            bleu_scores['bleu_score_1'].append(b_score_1)
            bleu_scores['bleu_score_2'].append(b_score_2)
            bleu_scores['bleu_score_3'].append(b_score_3)

    # 汇总各类指标得分
    exact_score = sum(exact_scores['hit']) / len(exact_scores['hit'])
    f1_score = sum(f1_scores['f1']) / len(f1_scores['f1'])
    precision = sum(f1_scores['precision']) / len(f1_scores['precision'])
    recall = sum(f1_scores['recall']) / len(f1_scores['recall'])

    bleu_score = sum(bleu_scores['bleu_score']) / len(bleu_scores['bleu_score'])
    bleu_score_1 = sum(bleu_scores['bleu_score_1']) / len(bleu_scores['bleu_score_1'])
    bleu_score_2 = sum(bleu_scores['bleu_score_2']) / len(bleu_scores['bleu_score_2'])
    bleu_score_3 = sum(bleu_scores['bleu_score_3']) / len(bleu_scores['bleu_score_3'])

    open_hit_score = sum(open_hit_scores['hit']) / len(open_hit_scores['hit'])
    closed_score = sum(closed_scores['hit']) / len(closed_scores['hit']) if len(closed_scores['hit']) != 0 else 0.0

    num_open, num_close = len(open_hit_scores['hit']), len(closed_scores['hit'])
    print(f'num_open {num_open} || num_close {num_close}')

    # 返回汇总结果
    return tabulate(
        [
            ['exact match score', exact_score * 100], 
            ['f1 score', f1_score * 100], 
            ['precision', precision * 100], 
            ['recall', recall * 100], 
            ['bleu_score', bleu_score * 100], 
            ['bleu_score_1', bleu_score_1 * 100], 
            ['bleu_score_2', bleu_score_2 * 100], 
            ['bleu_score_3', bleu_score_3 * 100], 
            ['open accuracy', open_hit_score * 100],
            ['yes/no accuracy', closed_score * 100]
        ], 
        headers=['Metric', 'Performance']
    )

if __name__ == '__main__':
    args = parse_option()

    dataset = args.gt.split("/")[-2]
    print(f"\n========\n {dataset}")

    gt = json.load(open(args.gt, 'r'))
    candidate = load_jsonl(args.candidate)
    pred = load_jsonl(args.pred)

    gt_ids = [item['image'] for item in gt]
    pred_ids = [item['question_id'] for item in pred]
    missing_in_pred = set(gt_ids) - set(pred_ids)
    extra_in_pred = set(pred_ids) - set(gt_ids)
    print(f"Missing in pred: {missing_in_pred}")
    print(f"Extra in pred: {extra_in_pred}")

    num_gt_ids, num_pred_ids = len(gt_ids), len(pred_ids)
    print(f'num_gt_ids: {num_gt_ids} || num_pred_ids: {num_pred_ids}')
    
    assert sorted(gt_ids) == sorted(pred_ids), "please make sure pred and gt are exactly matched"

    # 执行评估
    results = evaluate(gt, pred, candidate)
    print(results)
