# bert_joint

# required version
transformers 3.0.2

pytorch 1.7.0

# files
data preprocessing: prepare_nq_data.py

modeling and computing loss: model.py

initialization by pretrained SQuAD model: https://www.dropbox.com/s/8jnulb2l4v7ikir/model.zip

initialization, training process, prediction process, evaluation process: main.py

compute predictions: compute_predictions.py

evaluate precision, recall, F1 on predicted long answers and short answers: nq_eval.py

# run bert joint
python main.py
--data_dir (preprocessed training data directory)

--squad_model (BERT checkpoint trained on SQuAD 2.0)

--model (the directory to save/load the trained model)

--eval_data_dir (the original data for evaluation, non-preprocessed)

--eval_feature_dir (the preprocessed evaluation data directory)

--eval_result_dir (the directory to save predictions of the model on eval dataset)

--log_path (the directory to save the log file)
