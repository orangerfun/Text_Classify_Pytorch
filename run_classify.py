import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,6,7"

import time
import json
import torch
from tqdm import tqdm
from os.path import dirname, abspath
root_dir = dirname(abspath(__file__))
import sys
sys.path.append(root_dir)
import numpy as np
import pandas as pd
from finetuning_argparse import get_argparse
from tools.common import init_logger, logger, seed_everything
from callback.progressbar import ProgressBar
from processors.classify_processors import task_processors as processors
from transformers import WEIGHTS_NAME, BertConfig, get_linear_schedule_with_warmup, AdamW, BertTokenizer
from classify_model import BertForBinaryClassify
from build_dataset import load_and_cache_examples
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score


MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertForBinaryClassify, BertTokenizer),
}



def get_detail_output_info(file_path, save_path=None):
    '''
    通过predict函数得到输出的详细信息，包括输入，输出等
    file_path: 样本文件路径
    '''
    args, model, tokenizer = init()
    fp = open(file_path, "r", encoding="utf-8")
    lines = fp.readlines()
    result = {"all_text_a":[], "all_text_b":[], "all_label":[], "all_pred":[], "is_same":[]}
    all_label_ids, all_pred_ids = [], []
    for idx, l in enumerate(tqdm(lines, desc="predict")):
        if idx > 10000:
            logger.warning("选择前10000条测试程序, 若非测试程序请注释该判断语句！！")
            break
        l  = json.loads(l)
        text_a, text_b, label = l["question"], l.get("relationship", ""), l["label"]
        pred = predict(args, model, tokenizer, text_a, text_b)
        result["all_text_a"].append(text_a)
        result["all_text_b"].append(text_b)
        result["all_label"].append(label)
        result["all_pred"].append(pred)
        result["is_same"].append(label==pred)
        all_label_ids.append(args.label2id[label])
        all_pred_ids.append(args.label2id[pred])
    
    df = pd.DataFrame(result)
    if not save_path:
        save_dir = os.path.join(args.output_dir, "predict")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        tail = time.strftime('%Y%m%d%H%M%S', time.localtime())
        save_path = os.path.join(save_dir,"predict_result_"+tail+".xlsx")
        logger.info(f"未指定保存路径, 预测结果将保存在如下默认路径:{save_path}")
    df.to_excel(save_path, index=False)

    accuracy = accuracy_score(all_label_ids, all_pred_ids)
    if len(set(all_label_ids)) == 2:
        precision = precision_score(all_label_ids, all_pred_ids)
        recall = recall_score(all_label_ids, all_pred_ids)
    else:
        precision = precision_score(all_label_ids, all_pred_ids, average="macro")
        recall = recall_score(all_label_ids, all_pred_ids, average="macro")
    logger.info(f"accuracy:{accuracy}, precision:{precision}, recall:{recall}")
    logger.info("predict finished...")


def predict(args, model, tokenizer, text_a, text_b=""): 
    '''单条query推理'''       
    special_tokens_count = 3
    max_seq_length = args.train_max_seq_length
    text_a_tokens = tokenizer.tokenize(text_a)
    text_b_tokens = []

    if text_b:
        text_b_tokens = tokenizer.tokenize(text_b)
    if len(text_a_tokens)+len(text_b_tokens) > max_seq_length - special_tokens_count:
        text_a_tokens = text_a_tokens[: (max_seq_length - special_tokens_count-len(text_b_tokens))]
    if text_b:
        input_tokens = ["CLS"] + text_a_tokens+["SEP"] + text_b_tokens+["SEP"]
    else:
        input_tokens = ["CLS"]+text_a_tokens+["SEP"]

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    segment_ids = [0]*(len(text_a_tokens)+2)+[1]*(len(input_ids)-len(text_a_tokens)-2)
    input_mask = [1]*len(input_tokens)
    input_len = len(input_ids)
    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    padding_length = max_seq_length-len(input_ids)
    input_ids = input_ids+[pad_token]*padding_length
    input_mask = input_mask+[0]*padding_length
    segment_ids = segment_ids + [0]*padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    batch = ([input_ids], [input_mask], [segment_ids])
    batch = [torch.tensor(item).to(args.device) for item in batch]
    
    model.eval()
    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        if args.model_type != "distilbert":
            # XLM and RoBERTa don"t use segment_ids
            inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
        outputs = model(**inputs)
    logits = outputs[0]
    pred = np.argmax(logits.cpu().numpy(), axis=-1).tolist()[0]
    res = args.id2label[pred]
    return res


def evaluate(args, model, tokenizer, prefix="dev"):
    '''
    用于测试或者评估
    args: argparse实例
    model: 模型实例
    tokenizer: 分词器
    prefix: ["dev", "test"] dev 表示evalutate加载验证集, test是加载测试集
    '''
    eval_output_dir = os.path.join(args.output_dir, prefix)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, prefix)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # Eval!
    logger.info("***** Running %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc=prefix)

    results = {}
    all_preds, all_labels = [], []
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        preds = np.argmax(logits.cpu().numpy(), axis=-1).tolist()
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        all_preds += preds
        all_labels += out_label_ids
        pbar(step)

    eval_loss = eval_loss / nb_eval_steps
    accuracy = accuracy_score(all_labels, all_preds)
    # 二分类情况
    if len(set(all_labels))==2:
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
    # 多分类情况
    else:
        precision = precision_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")

    results['loss'] = [eval_loss]
    results["accuracy"] = [accuracy]
    results["precision"] = [precision]
    results["recall"] = [recall]

    logger.info("***** %s results *****", prefix)
    info = "-".join([f' {key}: {value[0]:.4f} ' for key, value in results.items()])
    logger.info(info)

    df = pd.DataFrame(results)
    tail = time.strftime('%Y%m%d%H%M%S', time.localtime())
    save_path = os.path.join(eval_output_dir, prefix+"_result_"+tail+".xlsx")
    df.to_excel(save_path, index=False)
    logger.info(f"{prefix} reuslt has been saved in {os.path.abspath(save_path)}")
    return results


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    # max_steps表示最大pao多少个step,若不指定直接根据数据集的数量计算得出
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # 多卡训练
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # 分布式训练
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    steps_trained_in_current_epoch = 0
    # TODO: Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # 从保存的checkpoint名字中读取总共跑的steps
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        # 计算执行了几个循环（epochs）
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        # 计算最后一个循环中跑了几个step
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    # 训练损失和打印的损失
    tr_loss, logging_loss = 0.0, 0.0
    # if args.do_adv:
    #     fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon)
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
    # save_step=-1:跑完一个epoch保存一次模型， logging_steps=1: 跑完一个epoch进行验证一次打印结果
    if args.save_steps==-1 and args.logging_steps==-1:
        args.logging_steps=len(train_dataloader)
        args.save_steps = len(train_dataloader)
    for epoch in range(int(args.num_train_epochs)):
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch)
        for step, batch in enumerate(train_dataloader):
            # 跳过已经训练过的steps
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[4].view(-1, 1)}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # 评估
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, model, tokenizer)
                # 保存模型： todo：根据domain classify 方式保存更多信息，如steps
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = (model.module if hasattr(model, "module") else model)
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    tokenizer.save_vocabulary(output_dir)
                    logger.info("Saving model checkpoint to %s"%(os.path.abspath(output_dir)))
                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # logger.info("Saving optimizer and scheduler states to %s", os.path.abspath(output_dir))
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def init():
    ''' 初始化：包括模型，参数等初始化'''
    args = get_argparse().parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) # mkdir
    args.output_dir = os.path.join(args.output_dir, args.model_type)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) # mkdir
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}_{args.task_name}.log')
    # 确定overwrite_output_dir参数是否合法：如果output_dir是空文件就不会被重写
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # 设置远程调试(如果需要)
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # 设置cuda, gpu 以及分布式训练: local_rank == -1 表示不用分布式训练
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
                "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank,device,args.n_gpu, bool(args.local_rank != -1),args.fp16,)
    # Set seed
    seed_everything(args.seed)
    
    args.task_name = args.task_name.lower()
    print("processors:",processors,";args.task_name:",args.task_name)
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    config.loss_type = args.loss_type
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case,)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("模型参数为 %s", args)
    return args, model, tokenizer
    


def main(params=[]):
    args, model, tokenizer = init()
    if params:
        args.do_train, args.do_test, args.do_predict = params

    # 1. 训练
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, "train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        print("SIM.do_train.completed.args.output_dir:"+str(args.output_dir))

    # 2. 测试
    if args.do_test:
        result = evaluate(args, model, tokenizer, "test")
    
    # 3. 推理
    if args.do_predict:
        while True:
            text_a = input("请输入text_a: ")
            text_b = input("请输入text_b: ")
            res = predict(args, model, tokenizer, text_a, text_b)
            print(res)



if __name__ == "__main__":
    # main中的参数指定是训练还是预测，默认（不指定）是训练, param=[训练，测试，推理]
    main()
    
    # 直接调用模型进行批量推理
    # file_path = "processed_data/THUCNews/test.json"
    # get_detail_output_info(file_path)