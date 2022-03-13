import os
import torch
from tqdm import tqdm
from os.path import abspath
from tools.common import logger
from torch.utils.data import TensorDataset
from processors.classify_processors import InputFeatures, task_processors as processors


def convert_examples_to_features(examples, tokenizer, label_list, max_seq_length):
    '''
    将example中的str转换成id, 构建input_id, attention_mask_id etc
    主要进行 token2id, add sepecial token, padding, truncating等操作
    '''
    
    features = []
    # label 2 id
    label_map = {l:idx for idx,l in enumerate(label_list)}
    for exp_idx, example in enumerate(tqdm(examples, desc="convert_features")):
        # 选取前20000条数据用于跑通程序，后续使用请注释掉
        if exp_idx > 20000:
            logger.warning("正在使用程序测试样例共2万条!!! 若非测试程序请注释该判断语句重跑代码")
            break
        query, text_b, task_type = example.text_a, example.text_b, example.task_type
        if query == "":            
            logger.info("text_a is none or empty string, this example pair will be skipped!")
            continue
        query_tokens = tokenizer.tokenize(query)

        if task_type == "sim":
            special_tokens_count = 3
            if text_b == "":
                logger.info("text_b is none or empty string, this example pair will be skipped!")
                continue
            textb_tokens = tokenizer.tokenize(text_b)
            # truncating
            # TODO: 截取方面需要进一步优化: 此处将text_a的末尾截取掉
            if len(query_tokens)+len(textb_tokens) > max_seq_length - special_tokens_count:
                query_tokens = query_tokens[: (max_seq_length - special_tokens_count-len(textb_tokens))]
            input_tokens = ["CLS"]+query_tokens+["SEP"]+textb_tokens+["SEP"]
        elif task_type == "classify":
            special_tokens_count = 2
            if len(query_tokens) > max_seq_length-special_tokens_count:
                query_tokens = query_tokens[:max_seq_length-special_tokens_count]
            input_tokens = ["CLS"]+query_tokens+["SEP"]
        else:
            raise ValueError("UNKONW TASK TYPE!!!")
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [0]*(len(query_tokens)+2)+[1]*(len(input_ids)-len(query_tokens)-2)
        input_mask = [1]*len(input_tokens)
        input_len = len(input_ids)
        # padding
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        padding_length = max_seq_length-len(input_ids)
        input_ids = input_ids+[pad_token]*padding_length
        input_mask = input_mask+[0]*padding_length
        segment_ids = segment_ids + [0]*padding_length
        label_id = label_map[example.labels]
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        if exp_idx < 3:
            logger.info("**feature example**")
            logger.info(f"text_a and text_b:{query} ## {text_b}")
            logger.info(f"input_ids: {input_ids}")
            logger.info(f"input_mask: {input_mask}")
            logger.info(f"segment_ids: {segment_ids}")
            logger.info(f"label_id: {label_id}")
        
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len, 
                                      segment_ids=segment_ids, label_ids=label_id))
    return features


def load_and_cache_examples(args, task, tokenizer, data_type):
    '''
    根据json文件构建数据集
    args: 参数实例
    task: 任务类型，classify:单条文本分类，sim:两条文本的相似度
    tokenizer: 分词器实例
    data_type: 数据类型，train:训练集， test:测试集，dev: 验证集，test_public: 其他测试集[后面会弃用] 
    return: tensordataset实例
    '''
    processor = processors[task]()
    # 处理好的数据保存路径, 命名方式:cached.{data_type}.{model_name}.{max_seq_len}.{task}
    cached_features_file = os.path.join(args.data_dir, 'cached.{}.{}.{}.{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type=='train' else args.eval_max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file: %s", abspath(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", abspath(args.data_dir))
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir, args.train_file_name, data_type)
        elif data_type == 'dev':    # evaluate 
            examples = processor.get_dev_examples(args.data_dir, args.dev_file_name, data_type)
        elif data_type == "test_public":
            examples = processor.get_test_pulic_examples(args.data_dir, args.test_public_file_name, data_type)
        elif data_type == "test":
            examples = processor.get_test_examples(args.data_dir, args.test_file_name, data_type)
        else:
            raise ValueError(f"Task not found `{data_type}` this type dataset!")
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type=='train' \
                                                               else args.eval_max_seq_length,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", os.path.abspath(cached_features_file))
            torch.save(features, cached_features_file)
    # if args.local_rank == 0 and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset