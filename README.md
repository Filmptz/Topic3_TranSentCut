# Introduction
We created a new Thai sentence segmentation model called *TranSentCut - Transformer Based Thai Sentence Segmentation* . Here you can find code for training/evaluating the model.

# Training the model

## Setup working directory
After cloning the repo, *cd* into it and create the following directories: 

* checkpoints
* data
* logs
* models
* tmp
* infer_result

## Get the training data, pretrained model and tokenizer
Go to https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased/tree/main. Download `config.json` and `pytorch_model.bin` and put them in `models/wangchanberta-base-att-spm-uncased`. Then download `tokenizer_config.json`, `sentencepiece.bpe.model` and `sentencepiece.bpe.vocab` and put them in `models/tokenizer`. Rename `tokenizer_config.json` to `config.json`. The `models` directory now should look like this

```
models/
    tokenizer/
        config.json <--- tokenizer_config.json renamed
        sentencepiece.bpe.model
        sentencepiece.bpe.vocab
    wangchanberta-base-att-spm-uncased/
        config.json
        pytorch_model.bin
```

## Train the model
Inside the container at `<container_path>` from earlier, run
```
python train.py --config_path=config/TranSentCutVersion2.yaml
```
The result will be written to `tmp/experiment_results.txt`. Model will be saved to `models/version2`. The training parameters in `config/TranSentCutVersion2.yaml` is the best configurations we found. It should give best f1-score (macro) of 0.0.9265. Space-correct should be 0.9626.

# Evaluate the model

Once the model finished training, it can be evaluated on new data using
```
python eval.py --model_path=models/version2 --tokenizer_path=models/tokenizer --eval_data_path=<eval_path> --context_length=256
```
where `<eval_path>` is the path to the evaluation data (.txt). 

The trained model is also available at https://drive.google.com/drive/folders/1G29LeCn4KiW5ZJZTLn-zNaNenTgIPCSu?usp=sharing if you just want to evaluate it. Replace `models/version2` in the above command with the path that you saved the model. Please get the tokenizer from https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased/tree/main. Then rename the tokenizer files and setup the working directory according to the instruction in the training section.

# Inference the model

Once the model finished training, it can be infered on new data using
```
python infer.py --model_path=models/version2 --tokenizer_path=models/tokenizer --infer_data_path=<data_infer_path> --context_length=256
```
where `<data_infer_path>` is the path to the inference data (.txt). The result will be saved to `infer_result/result_<data_infer_filename>.txt` 

# Reference 
(abstract only) TranSentCut − Transformer Based Thai Sentence Segmentation https://www.researchgate.net/publication/353996818_TranSentCut_-_Transformer_Based_Thai_Sentence_Segmentation. The full paper is under review.
# Team 
* Nartawat Phong-arom 6113059
* Jirawit Sopa 6113224
* Parichaya Thanawuthikrai 6113295