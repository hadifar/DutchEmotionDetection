# "xlm-roberta-base"
# "pdelobelle/robbert-v2-dutch-base"
# "GroNLP/bert-base-dutch-cased"
# "model_hub/xlm-roberta-base-ft-CSTwitter"
for lr in 4e-5 5e-5 6e-5
do
     python3 main.py --debug 0 --experiment bytime --crf 0 --task emotion --model_name xlm-roberta-base --lr $lr --epochs 5 --prev_history 0 --post_history 0 --train_data data/date_time/stratifiedComp_time_train.json --valid_data data/date_time/stratifiedComp_time_test.json
done

