python3 run_classify.py --task_name classify
                        --data_dir processed_data/THUCNews
                        --model_type bert
                        --model_name_or_path pretrained_model/chinese_rbt3_pytorch/
                        --output_dir outputs/THUCNews
                        --train_file_name train.json
                        --dev_file_name dev.json
                        --test_file_name test.json
                        --train_max_seq_length 512
                        --eval_max_seq_length 512
                        --do_train true
                        --do_predict false
                        --do_test false