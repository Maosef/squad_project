## Setup

1. Follow the steps 1 through 3 in [squad-main](squad-main\README.md) to setup the `squad` environment

2. Run `python load_bert.py`
    1. This loads in the pre-trained BERT model and saves it in the `squad-bert` directory

3. Run `python eval_bert.py --model_type=bert --model_name_or_path=squad-bert/ --output_dir=out/ --data_dir=squad-bert/ --predict_file=dev-v2.0.json`
    1. This evaluates the BERT model on the SQuAD dataset
    2. You can change `--output_dir` if you want to save the output to a different location
    3. Currently `dev-v2.0.json` is under the `squad-bert` directory, but if evaluating on a different file you can change `--data_dir` and `--predict_file`