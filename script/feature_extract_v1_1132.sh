cd ../
python3 coconut_train.py --features_extract  \
                         --train_set_file_path data/feature_extract_data_1132/train_set_v1_1_more_than_55.txt  \
                         --dev_set_file_path data/feature_extract_data_1132/dev_set_v1_1_more_than_55.txt  \
                         --idx_file_path data/feature_extract_data_1132/v1_idx_more_than_55.pickle \
                         --num_of_classes  1132 \
                         --model_version_string coconut_extract_model_v3 \
                         --batch_size 12 \
                         --lr 0.01       \
                         --epoch 600
                         --resume
