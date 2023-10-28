python3 main.py --debug 0 --experiment bytime_debug_eval --crf 0 --task emotion --model_name pdelobelle/robbert-v2-dutch-base --lr 5e-5 --epochs 5 --prev_history 0 --post_history 0 --train_data data/date_time/stratifiedComp_time_train.json --valid_data data/date_time/stratifiedComp_time_test.json

for mname in 'xlm-roberta-base' 'pdelobelle/robbert-v2-dutch-base' 'GroNLP/bert-base-dutch-cased'; do
  for task in 'response'; do

    python3 main.py --debug 0 --experiment bytime_v2 --crf 0 --task $task --model_name $mname --lr 5e-5 --epochs 5 --prev_history 0 --post_history 0 --train_data data/date_time/stratifiedComp_time_train.json --valid_data data/date_time/stratifiedComp_time_test.json

    python3 main.py --debug 0 --experiment bysector_airtele_v2 --crf 0 --task $task --model_name $mname --lr 5e-5 --epochs 5 --prev_history 0 --post_history 0 --train_data data/sector/train_airtele.json --valid_data data/sector/test_airtele.json
    python3 main.py --debug 0 --experiment bysector_pubtele_v2 --crf 0 --task $task --model_name $mname --lr 5e-5 --epochs 5 --prev_history 0 --post_history 0 --train_data data/sector/train_pubtele.json --valid_data data/sector/test_pubtele.json
    python3 main.py --debug 0 --experiment bysector_airpub_v2 --crf 0 --task $task --model_name $mname --lr 5e-5 --epochs 5 --prev_history 0 --post_history 0 --train_data data/sector/train_airpub.json --valid_data data/sector/test_airpub.json

    python3 main.py --debug 0 --experiment telecom1_v2 --crf 0 --task $task --model_name $mname --lr 5e-5 --epochs 5 --prev_history 0 --post_history 0 --train_data data/telecom/train1.json --valid_data data/telecom/test1.json
    python3 main.py --debug 0 --experiment telecom2_v2 --crf 0 --task $task --model_name $mname --lr 5e-5 --epochs 5 --prev_history 0 --post_history 0 --train_data data/telecom/train2.json --valid_data data/telecom/test2.json
    python3 main.py --debug 0 --experiment telecom3_v2 --crf 0 --task $task --model_name $mname --lr 5e-5 --epochs 5 --prev_history 0 --post_history 0 --train_data data/telecom/train3.json --valid_data data/telecom/test3.json
  done
done
