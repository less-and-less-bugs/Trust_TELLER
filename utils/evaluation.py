from typing import Optional, Union, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter

# def safe_completion(engine, prompt, MAX_TOKENS, stop, temp=0.0, logprobs=5, n = 1, num_tries = 0):
#     len_prompt_token = len(_TOKENIZER.tokenize(prompt))
#     if MAX_TOKENS + len_prompt_token >= GPT3_LENGTH_LIMIT:
#         print("OVERFLOW", MAX_TOKENS + len_prompt_token)
#         return {
#             "text": "overflow"
#         }
#     if n>1:
#         temp = 0.7
#     try:
#         resp = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=MAX_TOKENS, stop=stop,
#             temperature=temp, logprobs=logprobs, echo=True, n = n)
#     except Exception as e:
#         print(f'Encountered Error {e}, trying for the {num_tries} time.')
#         time.sleep(10)
#         if num_tries >= 10:
#             return None
#         else:
#             return safe_completion(engine, prompt, MAX_TOKENS, stop, temp, logprobs, \
#                 n, num_tries = num_tries + 1)
#     if n>1:
#         return resp
#     else:
#         return resp["choices"][0]
#
# def length_of_prompt(prompt, MAX_TOKENS, model='gpt3'):
#     if model == 'gpt3':
#         return len(_TOKENIZER.tokenize(prompt)) + MAX_TOKENS
#     elif model == 'gptj':
#         return len(_TOKENIZER_GPTJ.tokenize(prompt)) + MAX_TOKENS
#     else:
#         raise NotImplementedError(f'model {model} unimplemented')
#
# def in_context_manual_prediction(ex, training_data, engine, prompt_helper, model, new_rationale, length_test_only):
#     prompt, stop_signal = prompt_helper.prompt_for_answering_again(ex, training_data, new_rationale)
#     if length_test_only:
#         pred = length_of_prompt(prompt, _MAX_TOKENS)
#         return pred
#     elif model == 'gpt3':
#         pred = safe_completion(engine, prompt, _MAX_TOKENS, stop_signal, n = 1, temp=0.0, logprobs=5)
#         if pred != None:
#             if len(pred["text"]) > len(prompt):
#                 pred["text"] = pred["text"][len(prompt):]
#             else:
#                 pred["text"] = "null"
#             pred["completion_offset"] = len(prompt)
#     return pred
#
# def evaluate_manual_predictions(dev_set, verifying_qs, contexts, verifying_as, predictions, args, do_print=False):
#     acc_records = []
#     all_probs = []
#     all_texts = []
#
#     edited = 0
#     result_dict = {}
#     edited_correctly = 0
#     edited_falsely = 0
#     for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
#         gt = ex["label"]
#         id = ex['id']
#
#         p_ans = normalize_prediction(pred['answer'])
#         all_texts.append(p_ans)
#
#         acc = p_ans == gt
#         acc_records.append(acc)
#         all_probs.append(pred['answer_logprob'])
#         if do_print:
#             if pred['consistency'] < args.consistency_threshold:
#                 oa = pred['original_answer']
#                 acc_before = oa == gt
#                 print("--------------{} EX {} CONS {:.2f}--------------".format(id, acc, pred['consistency']))
#                 print('question: ', ex['question'])
#                 edited += 1
#                 vq = [c for c in verifying_qs if c['id']==id][0]['verifying_questions']
#                 cont = [c for c in contexts if c['id']==id][0]['context']
#                 va = [c for c in verifying_as if c['id']==id][0]['verifying_answers']
#                 try:
#                     print('original_rationale: ', pred['original_rationale'])
#                 except:
#                     print('original_rationale: ', 'none')
#                 print('original_answer: ', oa)
#                 sentences = rationale_tokenize(pred['original_rationale'])
#                 for j, (s, q, c, a) in enumerate(zip(sentences, vq, cont, va)):
#                     print('rationale_sentence {}: {}'.format(j, s))
#                     print('verifying_question {}: {}'.format(j, q))
#                     print('contexts {}: {}'.format(j, c))
#                     print('verifying_answers {}: {}'.format(j, a))
#                 print('P RAT:', pred['rationale'])
#                 print('P:', p_ans, 'G:', gt)
#                 if not acc:
#                     k = f'{oa}_to_{p_ans}_withgt_{gt}'
#                     print('k: ', k)
#                     if k in result_dict:
#                         result_dict[k] += 1
#                     else:
#                         result_dict[k] = 1
#                 if acc_before and (not acc):
#                     edited_falsely += 1
#                 elif (not acc_before) and acc:
#                     edited_correctly += 1
#     print('results: ')
#     for i in result_dict:
#         print(i, ': ', result_dict[i])
#     print(result_dict)
#     print(f'EDITED {edited} OUT OF {len(predictions)}')
#     print(f'Edited {edited_correctly} correctly and {edited_falsely} falsely')
#     print(f'{sum(acc_records)} correct out of {len(acc_records)}')
#     print("ACC", sum(acc_records) / len(acc_records))

def acc_compute(pt: list, gt: list)->float:
    correct = 0
    assert len(pt) == len(gt)
    total = len(pt)

    for p, g in zip(pt, gt):
        if p == g:
            correct += 1
    accuracy = correct / total
    return accuracy


def calculate_macro_f1(pt: list, gt: list)-> tuple[float, float, float]:
    unique_labels = set(gt)  # 获取唯一的标签值

    # 初始化统计指标的变量
    total_f1 = 0
    total_precision = 0
    total_recall = 0

    for label in unique_labels:
        # 计算每个类别的精确度、召回率和 F1 分数
        label_predicted = [p == label for p in pt]
        label_true = [t == label for t in gt]
        precision = precision_score(label_true, label_predicted)
        recall = recall_score(label_true, label_predicted)
        f1 = f1_score(label_true, label_predicted)
        print("Label: {} F1: {}, Pre: {}, Recall: {}".format(label, f1, precision, recall))
        # 累加总和
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # 计算宏平均 F1 分数
    macro_precision = total_precision / len(unique_labels)
    macro_recall = total_recall / len(unique_labels)
    macro_f1 = total_f1 / len(unique_labels)

    return macro_f1, macro_precision, macro_recall