# %%
import xml.etree.ElementTree as ET
import numpy as np
import random
import pickle
from tqdm import tqdm

from transformers import CamembertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import RobertaForSequenceClassification
from utils import WeightedTrainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score

from utils import genTrainingExamplesTrans
from sent_dataset import SentDataset
import argparse
import bios

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="the path to the yaml config file")
args = parser.parse_args()
params = bios.read(args.config_path)
# %%

with open(params["paths"]["positive_dataset_path"], 'rb') as f:
    positive_examples_orchid_news = pickle.load(f)

with open(params["paths"]["negative_dataset_path"], 'rb') as f:
    negative_examples_orchid_news = pickle.load(f)

print("positive (sb) examples",len(positive_examples_orchid_news),"negative (nsb) examples",len(negative_examples_orchid_news))

# %%
# train-test split
if params["training"]["balanced"]:
    negative_indices = [i for i in range(len(negative_examples_orchid_news))]
    random.shuffle(negative_indices)
    negative_indices = negative_indices[0:len(positive_examples_orchid_news)]
    y = [1 for i in range(len(positive_examples_orchid_news))] + [0 for i in range(len(positive_examples_orchid_news))]
else:
    negative_indices = [i for i in range(len(negative_examples_orchid_news))]
    y = [1 for i in range(len(positive_examples_orchid_news))] + [0 for i in range(len(negative_examples_orchid_news))]

X = positive_examples_orchid_news + [ negative_examples_orchid_news[i] for i in negative_indices ]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=params["training"]["seed"], shuffle=True)

# %%
# Tokenizer and dataset
tokenizer = CamembertTokenizer.from_pretrained(params["paths"]["tokenizer_path"])
train_dataset = SentDataset(X_train, y_train, tokenizer, context_length=params["model"]["context_length"])
test_dataset = SentDataset(X_test, y_test, tokenizer, context_length=params["model"]["context_length"])
cb = EarlyStoppingCallback(early_stopping_patience=5)

# %%
training_args = TrainingArguments(
    output_dir=params["paths"]["output_dir"],
    num_train_epochs=params["training"]["num_train_epochs"],
    per_device_train_batch_size=params["training"]["batch_size"],
    do_eval=True,
    logging_dir=params["paths"]["log_dir"],
    save_steps=200,
    save_total_limit=10,
    fp16=True,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model='f1-macro',
    greater_is_better=True,
    weight_decay=params["training"]["weight_decay"],
    eval_steps=200,
    learning_rate=params["training"]["learning_rate"],
    seed=params["training"]["seed"]
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1) # 1D
    sc = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=["I", "E"], output_dict=True, digits=4)
    f1 = f1_score(labels, predictions, average="macro")
    
    return {
        'f1-macro':f1,
        'space-correct':sc,
        'I-precision': report["I"]["precision"],
        'I-recall': report["I"]["recall"],
        "I-f1": report["I"]["f1-score"],
        'E-precision': report["E"]["precision"],
        'E-recall': report["E"]["recall"],
        "E-f1": report["E"]["f1-score"],    
    }

def init_model():
    return RobertaForSequenceClassification.from_pretrained(params["paths"]["pretrain_path"], num_labels=2)

if params["training"]["balanced"]:
    trainer = Trainer(
        model_init=init_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[cb]
    )
else:
    class_counts = np.array([len(negative_examples_orchid_news),len(positive_examples_orchid_news)])
    class_weights = np.array([1/(len(negative_examples_orchid_news)/len(positive_examples_orchid_news)),1])
    trainer = WeightedTrainer(
        model_init=init_model, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[cb],
        class_weights=class_weights
    )

# %%
trainer.train()
print(params)
print(trainer.evaluate())
print("----------------------------------------------------")

with open("tmp/experiment_results_version2.txt", "a+") as f:
    print(params, file=f)
    print(trainer.evaluate(), file=f)
    print("----------------------------------------------------", file=f)

# %%
trainer.save_model(params["paths"]["save_dir"])
