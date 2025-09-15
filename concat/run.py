from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel)
from tqdm import tqdm, trange
import multiprocessing
from JITFine.concat.model import Model
from JITFine.my_util import convert_examples_to_features, TextDataset, eval_result, preprocess_code_line, \
    get_line_level_metrics, create_path_if_not_exist

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 5
    args.warmup_steps = 0
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    best_f1 = 0
    global_step = 0
    model.zero_grad()
    patience = 0

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
            model.train()
            # loss, logits, _ = model(inputs_ids, attn_masks, manual_features, labels)
            loss, logits, _, _ = model(inputs_ids, attn_masks, manual_features, labels)
            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % args.save_steps == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(tr_loss / tr_num, 5)))
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1
            if (step + 1) % args.save_steps == 0:
                results = evaluate(args, model, tokenizer, eval_when_training=True)
                checkpoint_prefix = f'epoch_{idx}_step_{step}'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                torch.save({
                    'epoch': idx,
                    'step': step,
                    'patience': patience,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}, output_dir)
                logger.info("Saving epoch %d step %d model checkpoint to %s, patience %d", idx, global_step, output_dir,
                            patience)
                # Save model checkpoint
                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1']
                    logger.info("  " + "*" * 20)
                    logger.info("  Best f1:%s", round(best_f1, 4))
                    logger.info("  " + "*" * 20)

                    checkpoint_prefix = 'checkpoint-best-f1'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                    patience = 0
                    torch.save({
                        'epoch': idx,
                        'step': step,
                        'patience': patience,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                else:
                    patience += 1
                    if patience > args.patience * 5:
                        logger.info('patience greater than {}, early stop!'.format(args.patience))
                        return


def evaluate(args, model, tokenizer, eval_when_training=False):
    # build dataloader
    logger.info("Building evaluation dataset without caching")
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file, mode='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            # loss, logit, _ = model(inputs_ids, attn_masks, manual_features, labels)
            loss, logit,_,_ = model(inputs_ids, attn_masks, manual_features, labels)
            torch.cuda.empty_cache()
            eval_loss += loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    y_preds = logits[:, -1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='binary')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, tokenizer, best_threshold=0.5):
    # build dataloader
    logger.info("Building test dataset without caching")
    eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file, mode='test')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)
    # ➕ 为 model 设置 msg/code 区间（供 forward 引导用）
    model.msg_ranges = [ex.msg_range for ex in eval_dataset.examples]
    model.code_ranges = [ex.code_range for ex in eval_dataset.examples]

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    attns = []
    for batch in eval_dataloader:
        (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, prob, cls2tok_attn, guide_attn = model(inputs_ids, attn_masks, manual_features, labels,
                                                         output_attentions=True)

            #loss, logit, attn_weights = model(inputs_ids, attn_masks, manual_features, labels, output_attentions=True)

            eval_loss += loss.mean().item()
            logits.append(prob.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            # attns.append(last_layer_attn_weights.cpu().numpy())
            attns.append((cls2tok_attn.detach().cpu(), guide_attn.detach().cpu()))  # ✅ 保存成 tuple，支持后续 msg2code  # 保持是 torch.Tensor
            # attns.append(last_layer_attn_weights)  # ❗保留 torch.Tensor，不转 numpy

        nb_eval_steps += 1
    # output result
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    #attns = np.concatenate(attns, 0)

    y_preds = logits[:, -1] > best_threshold

    # 🔥 记录 attn_vec 对应变更段（热力图数据）
    heatmap_data = []
    heatmap_labels = []

    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='binary')

    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    result = []

    # Remove caching of buggy lines and reload every time
    logger.info("Loading buggy line dataset without caching")
    commit2codes, idx2label = commit_with_codes(args.buggy_line_filepath, tokenizer)

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = [], [], [], [], []


    # for example, pred, prob, attn in zip(eval_dataset.examples, y_preds, logits[:, -1], attns):
    #     result.append([example.commit_id, prob, pred, example.label])
    for idx, (example, pred, prob, attn_pair) in enumerate(zip(eval_dataset.examples, y_preds, logits[:, -1], attns)):
        result.append([example.commit_id, prob, pred, example.label])

        # attn_pair 是 tuple: (cls2tok_attn, guide_attn)
        if isinstance(attn_pair, tuple):
            cls2tok_attn, guide_attn = attn_pair
        else:
            cls2tok_attn = attn_pair
            guide_attn = None

        if int(example.label) == 1 and int(pred) == 1 and '[ADD]' in example.input_tokens:
            cur_codes = commit2codes[commit2codes['commit_id'] == example.commit_id]
            cur_labels = idx2label[idx2label['commit_id'] == example.commit_id]

            # === 使用 msg→code 引导 attention 替代 cls2tok ===
            if guide_attn is not None:
                # ✅ example.msg_range / code_range 是 tuple，例如 (1,20), (20,350)
                msg_start, msg_end = example.msg_range
                code_start, code_end = example.code_range

                if isinstance(guide_attn, torch.Tensor):
                    #attn_map = guide_attn[idx]  # [seq_len, seq_len]
                    attn_map = guide_attn.squeeze(0)  # or guide_attn[0] if batch_size=
                else:
                    attn_map = torch.tensor(guide_attn.squeeze(0))  # 转 tensor

                msg_to_code = attn_map[msg_start:msg_end, code_start:code_end]  # [msg_len, code_len]
                attn_vec = msg_to_code.mean(dim=0)  # ➜ 每个 code token 被 msg 的平均关注程度
            else:
                # fallback 使用 cls→token
                if isinstance(cls2tok_attn, torch.Tensor):
                    attn_vec = cls2tok_attn.squeeze(0)
                else:
                    attn_vec = torch.tensor(cls2tok_attn.squeeze(0))

                    # numpy 化
            attn_vec = attn_vec.detach().cpu().numpy()


            # # 收集热力图 attention scores（仅 ADD ~ DEL 区间）
            # tokens = example.input_tokens
            # try:
            #     begin_pos = tokens.index('[ADD]')
            #     end_pos = tokens.index('[DEL]') if '[DEL]' in tokens else len(tokens)
            #
            #     if end_pos > begin_pos:
            #         region_attn = attn_vec[begin_pos:end_pos]  # attention 分数
            #         region_tokens = tokens[begin_pos:end_pos]  # 真正的 token 列表
            #         heatmap_data.append({
            #             "attn": region_attn,  # numpy.ndarray or list of floats
            #             "tokens": region_tokens,  # list of token strings
            #             "commit_id": example.commit_id,
            #             "msg_range": example.msg_range,
            #             "code_range": example.code_range
            #         })
            #         heatmap_labels.append(example.commit_id)
            # except Exception as e:
            #     logger.warning(f"[WARN] Cannot extract attention region: {example.commit_id}")
            #     continue

            # cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, cur_top_10_acc, cur_top_5_acc = deal_with_attns(
            #     example, attn_vec,
            #     pred, cur_codes,
            #     cur_labels, args.only_adds)
                # ✅ 调用增强版 deal_with_attns
            line_df, cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, cur_top_10_acc, cur_top_5_acc = deal_with_attns(
                    example, attn_vec, pred, cur_codes, cur_labels, args.only_adds, return_df=True
                )
            # 保存当前 commit 的行级结果 CSV
            commit_csv_path = os.path.join(args.output_dir, f"{example.commit_id}_line_scores.csv")
            line_df.to_csv(commit_csv_path, sep='\t', index=False)

            # 写入 log 文件
            logger.info(f"Saved {len(line_df)} lines for commit {example.commit_id} -> {commit_csv_path}")


            IFA.append(cur_IFA)
            top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
            effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
            top_10_acc.append(cur_top_10_acc)
            top_5_acc.append(cur_top_5_acc)

    logger.info(
        'Top-10-ACC: {:.4f},Top-5-ACC: {:.4f}, Recall20%Effort: {:.4f}, Effort@20%LOC: {:.4f}, IFA: {:.4f}'.format(
            round(np.mean(top_10_acc), 4), round(np.mean(top_5_acc), 4),
            round(np.mean(top_20_percent_LOC_recall), 4),
            round(np.mean(effort_at_20_percent_LOC_recall), 4), round(np.mean(IFA), 4))
    )
    RF_result = pd.DataFrame(result)
    # RF_result.to_csv(os.path.join(args.result_output_dir, "predictions.csv"), sep='\t', index=None)
    RF_result.to_csv(os.path.join(args.output_dir, "predictions.csv"), sep='\t', index=None)
    # ✅ 保存热力图 attention scores
    # with open(os.path.join(args.output_dir, f"heatmap_{args.attn_fusion}_head{args.head_fusion}.pkl"), 'wb') as f:
    #     pickle.dump({"data": heatmap_data, "labels": heatmap_labels}, f)
    # with open("heatmap_guide_raw.pkl", "wb") as f:
    #     pickle.dump(heatmap_data, f)

def commit_with_codes(filepath, tokenizer):
    data = pd.read_pickle(filepath)
    commit2codes = []
    idx2label = []
    for _, item in data.iterrows():
        commit_id, idx, changed_type, label, raw_changed_line, changed_line = item
        line_tokens = [token.replace('\u0120', '') for token in tokenizer.tokenize(changed_line)]
        for token in line_tokens:
            commit2codes.append([commit_id, idx, changed_type, token])
        idx2label.append([commit_id, idx, label])
    commit2codes = pd.DataFrame(commit2codes, columns=['commit_id', 'idx', 'changed_type', 'token'])
    idx2label = pd.DataFrame(idx2label, columns=['commit_id', 'idx', 'label'])
    return commit2codes, idx2label


def deal_with_attns(item, attn_vec, pred, commit2codes, idx2label, only_adds=False,return_df=False):
    '''
    score for each token
    :param item:
    :param attns:
    :param pred:
    :param commit2codes:
    :param idx2label:
    :return:
    '''
    commit_id = item.commit_id
    input_tokens = item.input_tokens
    commit_label = item.label

    # cls2tok_attn, guide_attn = attn_vec  # ✅ 新版 tuple
    # cls2tok_attn = cls2tok_attn.numpy() if isinstance(cls2tok_attn, torch.Tensor) else cls2tok_attn
    # guide_attn = guide_attn.numpy() if isinstance(guide_attn, torch.Tensor) else guide_attn

    # 🎯 不再用 input_tokens[ADD:DEL] 切 token，而是直接对应 code_range 区间
    code_start, code_end = item.code_range
    code_tokens = input_tokens[code_start:code_end]

    # ✅ 强制截断对齐
    if len(attn_vec) > len(code_tokens):
        attn_vec = attn_vec[:len(code_tokens)]
    elif len(code_tokens) > len(attn_vec):
        code_tokens = code_tokens[:len(attn_vec)]


    # # remove msg,cls,eos,del
    # begin_pos = input_tokens.index('[ADD]')
    # end_pos = input_tokens.index('[DEL]') if '[DEL]' in input_tokens else len(input_tokens) - 1
    #
    # # === Step 2：处理 attention 向量 ===
    # if isinstance(attn_vec, tuple):
    #     # 新版传入的是 (cls2tok_attn, guide_attn)
    #     _, guide_attn = attn_vec
    #     if isinstance(guide_attn, torch.Tensor):
    #         guide_attn = guide_attn.detach().cpu().numpy()
    # else:
    #     guide_attn = attn_vec  # 若直接传的是向量
    #
    #     # === Step 3：裁剪变更 token 范围（注意防止越界）===
    # guide_attn = guide_attn[:len(input_tokens)]  # 防止越界
    # tokens = input_tokens[begin_pos:end_pos]
    # scores = guide_attn[begin_pos:end_pos]
    #
    # if len(tokens) != len(scores):
    #     print(f"⚠️ [Mismatch] tokens: {len(tokens)}, scores: {len(scores)} → {item.commit_id}")
    #     return 0, 0, 0, 0, 0

    #attn_vec = attn_vec[:len(input_tokens)]  # 防止越界

    # tokens = input_tokens[begin_pos:end_pos]
    # scores = guide_attn

    attn_df = pd.DataFrame({
        'token': [tok.replace('\u0120', '') for tok in code_tokens],
        'score': attn_vec
    })

    # attns = attns.mean(axis=0)[begin_pos:end_pos]
    # attn_df['score'] = attns
    attn_df = attn_df.sort_values(by='score', ascending=False)
    attn_df = attn_df.groupby('token').sum()
    attn_df['token'] = attn_df.index
    attn_df = attn_df.reset_index(drop=True)

    # calculate score for each line in commit
    if only_adds:
        commit2codes = commit2codes[commit2codes['changed_type'] == 'added']  # only count for added lines
    commit2codes = commit2codes.drop('commit_id', axis=1)
    commit2codes = commit2codes.drop('changed_type', axis=1)

    result_df = pd.merge(commit2codes, attn_df, how='left', on='token')
    #修改1
    # result_df = result_df.groupby(['idx']).sum()
    # result_df = result_df.reset_index(drop=False)
    result_df = result_df.groupby(['idx']).agg(
        total_tokens=('token', 'count'),
        line_score=('score', 'sum')
    ).reset_index()

    result_df = pd.merge(result_df, idx2label, how='inner', on='idx')
    #修改2add
    result_df = result_df.rename(columns={'label': 'line_level_label', 'idx': 'row'})

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = get_line_level_metrics(
        result_df['line_score'].tolist(), result_df['line_level_label'].tolist())

    if return_df:
        return result_df, IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc
    else:
        return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc
    #return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", nargs=2, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--result_output_dir", default=None, type=str, required=True,
    #                     help="The result_output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--eval_data_file", nargs=2, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", nargs=2, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_seed', type=int, default=123456,
                        help="random seed for data order initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    parser.add_argument('--feature_size', type=int, default=14,
                        help="Number of features")
    parser.add_argument('--num_labels', type=int, default=2,
                        help="Number of labels")
    parser.add_argument('--semantic_checkpoint', type=str, default=None,
                        help="Best checkpoint for semantic feature")
    parser.add_argument('--manual_checkpoint', type=str, default=None,
                        help="Best checkpoint for manual feature")
    parser.add_argument('--max_msg_length', type=int, default=64,
                        help="Number of labels")
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stop')
    parser.add_argument("--only_adds", action='store_true',
                        help="Whether to run eval on the only added lines.")
    parser.add_argument("--buggy_line_filepath", type=str,
                        help="complete buggy line-level  data file for RQ3")
    parser.add_argument("--head_dropout_prob", type=float, default=0.1,
                        help="Dropout probability for head layers")
    parser.add_argument("--no_abstraction", action='store_true',
                        help="Disable abstraction in code processing")
    # parser.add_argument("--attn_fusion", type=str, choices=["last", "last2", "last4", "all", "layer10", "layer8"],
    #                     default="last2",
    #                     help="How to fuse attention layers: 'last2', 'last4', or 'all'")
    # parser.add_argument("--head_fusion", default="all", type=str,
    #                     help="Which heads to fuse in the last layer: all, or indices like 0,1,2")
    # parser.add_argument("--head_weights",type=str,default=None,
    #                     help="逗号分隔的 12 个 attention head 权重，例如 '0.05,0.1,...'，用于加权融合各个 head 分数")
    # parser.add_argument("--head_ids", type=str, default=None,
    #                     help="Comma-separated list of attention head indices (used when head_fusion == 'weighted')")
    parser.add_argument("--learn_head_weights", action="store_true",
                        help="是否训练可学习的 head 融合权重")

    args = parser.parse_args()
    return args


def main(args):
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )
    # Set seed

    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = args.num_labels
    config.feature_size = args.feature_size
    config.hidden_dropout_prob = args.head_dropout_prob
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    special_tokens_dict = {'additional_special_tokens': ["[ADD]", "[DEL]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = RobertaModel.from_pretrained(args.model_name_or_path, config=config)

    model.resize_token_embeddings(len(tokenizer))
    logger.info("Training/evaluation parameters %s", args)

    model = Model(model, config, tokenizer, args)
    # Training
    if args.do_train:
        if args.semantic_checkpoint:
            semantic_checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.semantic_checkpoint, '{}'.format(semantic_checkpoint_prefix))
            logger.info("Loading semantic checkpoint from {}".format(output_dir))
            checkpoint = torch.load(output_dir)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if args.manual_checkpoint:
            manual_checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.manual_checkpoint, '{}'.format(manual_checkpoint_prefix))
            logger.info("Loading manual checkpoint from {}".format(output_dir))
            checkpoint = torch.load(output_dir)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file)

        for idx, example in enumerate(train_dataset.examples[:1]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("label: {}".format(example.label))
            logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
            logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        checkpoint = torch.load(output_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        result = evaluate(args, model, tokenizer)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        checkpoint = torch.load(output_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Successfully load epoch {}'s model checkpoint".format(checkpoint['epoch']))
        model.to(args.device)
        test(args, model, tokenizer, best_threshold=0.5)
        eval_result(os.path.join(args.output_dir, "predictions.csv"), args.test_data_file[-1])

    return results


if __name__ == "__main__":
    cur_args = parse_args()
    create_path_if_not_exist(cur_args.output_dir)
    main(cur_args)
