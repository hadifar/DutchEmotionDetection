for task in 'valence'
#for task in 'response'
#for task in 'emotion'task in ;
do
    echo $task
#    python3 baseline_classical.py --debug 0 --experiment bytime_v100 --task $task  --model_name svm --prev_history 0 --post_history 0 --train_data data/date_time/stratifiedComp_time_train.json --valid_data data/date_time/stratifiedComp_time_test.json

    python3 baseline_classical.py --debug 0 --experiment bysector_airpub --task $task --model_name svm --prev_history 0 --post_history 0 --train_data data/sector/train_airpub.json --valid_data data/sector/test_airpub.json
    python3 baseline_classical.py --debug 0 --experiment bysector_pubtele --task $task  --model_name svm --prev_history 0 --post_history 0 --train_data data/sector/train_pubtele.json --valid_data data/sector/test_pubtele.json
    python3 baseline_classical.py --debug 0 --experiment bysector_airtele --task $task  --model_name svm --prev_history 0 --post_history 0 --train_data data/sector/train_airtele.json --valid_data data/sector/test_airtele.json

    echo '------------------------'
    echo '------------------------'
#    python3 baseline_classical.py --debug 0 --experiment telecom1 --task $task  --model_name svm --prev_history 0 --post_history 0 --train_data data/telecom/train1.json --valid_data data/telecom/test1.json
#    python3 baseline_classical.py --debug 0 --experiment telecom2 --task $task  --model_name svm --prev_history 0 --post_history 0 --train_data data/telecom/train2.json --valid_data data/telecom/test2.json
#    python3 baseline_classical.py --debug 0 --experiment telecom3 --task $task  --model_name svm --prev_history 0 --post_history 0 --train_data data/telecom/train3.json --valid_data data/telecom/test3.json
done
