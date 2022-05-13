CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port 13117 --nproc_per_node=1 \
       Trainer.py --do_train --do_eval --do_predict --evaluate_during_training \
                  --data_dir="data/dataset/NER/weibo" \
                  --output_dir="data/result/NER/weibo/WCBertCRF_Token_add_1" \
				  --logging_dir="data/log/NER/weibo/WCBertCRF_Token_add_1" \
                  --label_file="data/dataset/NER/weibo/labels.txt" \
                  --saved_embedding_dir="data/dataset/NER/weibo" \
                  --model_type="WCBertCRF_Token" \
				  --fusion_method="add" \
				  --default_label="O" \
				  --max_scan_num=1500000 \
                  --max_word_num=5 \
                  --seed=106524 \
                  --per_gpu_train_batch_size=4 \
                  --per_gpu_eval_batch_size=16 \
                  --learning_rate=1e-5 \
                  --max_steps=-1 \
                  --max_seq_length=256 \
                  --num_train_epochs=20 \
                  --warmup_steps=190 \
                  --save_steps=600 \
                  --logging_steps=100\
				  --config_name="data/berts/bert/config.json" \
                  --model_name_or_path="data/berts/bert/pytorch_model.bin" \
                  --vocab_file="data/berts/bert/vocab.txt" \
                  --word_vocab_file="data/vocab/tencent_vocab.txt" \
				  --word_embedding="data/embedding/word_embedding.txt"

