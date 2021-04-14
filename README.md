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

