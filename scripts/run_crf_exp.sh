# "xlm-roberta-base"
# "pdelobelle/robbert-v2-dutch-base"
# "GroNLP/bert-base-dutch-cased"
for mname in 'pdelobelle/robbert-v2-dutch-base'; do
  for task in 'emotion'; do
    for seed in 0; do
      python3 main.py --debug 0 --seed $seed --experiment crf_exp_bytime --crf 0 --task $task --model_name $mname --lr 5e-5 --epochs 5 --prev_history 0 --post_history 0 --train_data data/date_time/stratifiedComp_time_train.json --valid_data data/date_time/stratifiedComp_time_test.json
      python3 main.py --debug 0 --seed $seed --experiment crf_exp_bytime --crf 1 --task $task --model_name $mname --lr 5e-5 --epochs 5 --prev_history 0 --post_history 0 --train_data data/date_time/stratifiedComp_time_train.json --valid_data data/date_time/stratifiedComp_time_test.json
    done
  done
done
