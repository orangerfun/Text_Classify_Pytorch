import argparse

def get_argparse():
     parser = argparse.ArgumentParser()

     # Required parameters
     task_name = "classify"
     pretrained = "pretrained_model/chinese_rbt3_pytorch/"
     saved = "outputs/sim_output/bert/checkpoint-3376"
     parser.add_argument("--task_name", default=task_name, type=str, choices=["classify", "sim"]
                         help="任务类型, classify: 单条文本分类, sim: 两条文本相似度 ")
     parser.add_argument("--data_dir", default="processed_data/THUCNews", type=str,
                         help="训练测试验证数据集所在目录", )
     parser.add_argument("--model_type", default="bert", type=str,
                         help="模型类型当前仅可选bert")
     parser.add_argument("--model_name_or_path", default=pretrained, type=str,
                         help="预训练模型所在目录" )
     parser.add_argument("--output_dir", default=f"outputs/THUCNews", type=str, 
                         help="输出目录(保存预测结果和模型)", )
     # parser.add_argument("--data_type", default="train", type=str, choices=["train", "test", "dev", "test_pulic"],
     #                     help="数据类型(用于指定处理训练/测试/验证数据)", )
     parser.add_argument("--train_file_name", default="train.json", type=str,
                         help="训练集文件名称", )
     parser.add_argument("--dev_file_name", default="dev.json", type=str,
                         help="验证集文件名称", )
     parser.add_argument("--test_file_name", default="test.json", type=str,
                         help="测试集文件名称", )
     parser.add_argument("--train_max_seq_length", default=512, type=int,
                         help= "训练集最大长度")
     parser.add_argument("--eval_max_seq_length", default=512, type=int,
                         help="验证数据最大长度", )
     parser.add_argument("--do_train", default=True, type=bool,
                         help="是否训练")
     parser.add_argument("--do_predict", default=False, type=bool,
                         help="是否要推理")
     parser.add_argument("--do_test", default=False, type=bool,
                         help="是否要测试")

     # Other parameters
     parser.add_argument('--loss_type', default='ce', type=str,
                         choices=['lsr', 'focal', 'ce'])
     parser.add_argument("--config_name", default="", type=str,
                         help="Pretrained config name or path if not the same as model_name")
     parser.add_argument("--tokenizer_name", default="", type=str,
                         help="Pretrained tokenizer name or path if not the same as model_name", )
     parser.add_argument("--cache_dir", default="", type=str,
                         help="Where do you want to store the pre-trained models downloaded from s3", )
     parser.add_argument("--evaluate_during_training", action="store_true",
                         help="Whether to run evaluation during training at each logging step.", )
     parser.add_argument("--do_lower_case", default=True, type=bool,
                         help="Set this flag if you are using an uncased model.")
     # adversarial training
     parser.add_argument("--do_adv", action="store_true",
                         help="Whether to adversarial training.")
     parser.add_argument('--adv_epsilon', default=1.0, type=float,
                         help="Epsilon for adversarial.")
     parser.add_argument('--adv_name', default='word_embeddings', type=str,
                         help="name for adversarial layer.")
     parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                         help="Batch size per GPU/CPU for training.")
     parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                         help="Batch size per GPU/CPU for evaluation or test.")
     parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                         help="Number of updates steps to accumulate before performing a backward/update pass.", )
     parser.add_argument("--learning_rate", default=3e-5, type=float,
                         help="The initial learning rate for Adam.")
     parser.add_argument("--weight_decay", default=0.01, type=float,
                         help="Weight decay if we apply some.")
     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                         help="Epsilon for Adam optimizer.")
     parser.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     parser.add_argument("--num_train_epochs", default=2, type=float,
                         help="Total number of training epochs to perform.")
     parser.add_argument("--max_steps", default=-1, type=int,
                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
     parser.add_argument("--warmup_proportion", default=0.1, type=float,
                         help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
     parser.add_argument("--logging_steps", type=int, default=-1,
                         help="Log every X updates steps.")
     parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every X updates steps.")
     parser.add_argument("--eval_all_checkpoints", action="store_true",
                         help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
     parser.add_argument("--predict_checkpoints",type=int, default=0,
                         help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
     parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
     parser.add_argument("--overwrite_output_dir", default=True, type=bool,
                         help="Overwrite the content of the output directory")
     parser.add_argument("--overwrite_cache", action="store_true",
                         help="Overwrite the cached training and evaluation sets")
     parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
     parser.add_argument("--fp16", action="store_true",
                         help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
     parser.add_argument("--fp16_opt_level", type=str, default="O1",
                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                              "See details at https://nvidia.github.io/apex/amp.html", )
     parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
     parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
     parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
     return parser