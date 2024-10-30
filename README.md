<p align="center">
  <img src="./assets/repository-profile.jpg" />
</p>
<!-- ![Profile Image](./assets/repository-profile.jpg) -->

# arabic-nano-gpt

Arabic Nano GPT Trained on Arabic Wikipedia Dataset from Wikimedia. This code is for education and demonstration purposes to experience the entire workflow of LLMs **pre-training** on the Nano Scale. This code is designed to load a dataset, preprocess its text, train a tokenizer on it, and lastly train a model using _Causal Language Modeling_.

|     Model Name     | Parameters | Size  |
| :----------------: | :--------: | :---: |
| arabic-nano-gpt-v0 |    5 M     | 26 MB |
| arabic-nano-gpt-v0 |    10 M    | 40 MB |
| arabic-nano-gpt-v0 |    20 M    | 90 MB |

## Setup

This environment is setup to work on a Linux platform. Make sure to use WSL2 on windows.

- Clone this repository.

```bash
git clone https://github.com/e-hossam96/arabic-nano-gpt.git
```

- Install developer tools for C++ package building. _used for torch compile_

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install build-essential
```

- Download and install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda
```

- Activate conda `base` environment.

```bash
source ./miniconda/bin/activate
```

- Create **nanogpt** env from [YAML](./environment.yml) file

```bash
cd arabic-nano-gpt
conda env create -f environment.yml
conda activate nanogpt
```

- Fill [example.env](./example.env) file and save it into a `.env` file.

```bash
cp example.env .env
```

## Execution

The _bash_ script code in [train_ml.sh](./train_lm.sh) is used to define the configurations to run all the Python scripts in the `src` folder. All you need to do is to define your configurations based on your needs and run the bash script and it handle every thing as follows.

- Decide on the _raw_ data and where its preprocessed version will be saved.

```bash
DATA_CKPT=wikimedia/wikipedia
SUB_DATA=20231101.ar
PROCESSED_DATA_PATH=data/$DATA_CKPT.csv
```

- Define the tokenizer configurations. The default is a `GPT2` based tokenizer trained from scratch on the data. After training, the tokenizer will be saved in the same directory as your intended model.

```bash
BASE_MODEL=openai-community/gpt2
MODEL_NAME=arabic-nano-gpt-v2
MODEL_PATH=models/$MODEL_NAME
MODEL_MAX_LENGTH=1024
VOCAB_SIZE=16384
```

- Define the end model configurations. The more the parameters, the larger the model and the longer it needs to train on the data.

```bash
EMBED_SIZE=384
NUM_ATT_HEAD=6
NUM_ATT_LAYERS=8
```

- Define the training parameters. Make sure to choose reasonable values. The default is to train the model on the entire data but this will take so long. You can define another parameter called `SPLIT_SIZE` and use it as input to the [train_causeal_lm.py](./src/train_causal_lm.py) to select a small sample of the data.

```bash
NUM_EPOCHS=8
BATCH_SIZE=32
ACCUM_STEPS=8
EVAL_STEPS=5000
LOG_STEPS=2000
LR=0.0001
WD=0.000001
WARMUP=0.01
```

- Run the script inside the activated `nanogpt` conda environment and wait tell the results are logged on your **Weights & Biased** account.

```bash
bash train_lm.sh
```

## Small Batch Overfitting

Before training your model for a long time and not being sure about its final state, you should experiment with a small batch using the `SPLIT_SIZE` parameter and try to bring the training loss to near zero. In the three checkpoints we have produced, we ensured that the architecture and the training step will actually bring the loss to near zero. W&B runs will be shared later to check the values.

To do so, you need to define a reasonable learning rate (`LR`) for the small batch (`SPLIT_SIZE`) and train for longer using the `NUM_EPOCHS`. You should also comment the **early stopping** `callback` from the HuggingFace's `Trainer` in this step in [train_causeal_lm.py](./src/train_causal_lm.py).

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)
```

Once validated, you can remove the `SPLIT_SIZE` parameter, re-define the training configurations to match the full data training, and run the codes safely.

## Data Pre-processing

For this step, we followed simple steps to preprocess and clean the text. The dataset is already almost clean but we needed to further preprocess it before training. We extracted all the paragraphs by splitting on the `\n\n` characters after each paragraph in an article. Also, we removed all the diacritics using the `strip_tashkeel` function from `PyAraby` and padded all the punctuations using white spaces as per the **AraGPT2** paper. This left us with around **8.5 Million** paragraphs with the following length distribution.

![Length Distribution of Original Paragraphs](./assets/length-dist-all-docs.png)

We further removed all the sentences that are less than _60_ and more than _1250_ characters to have consistent-length paragraphs (docs from now on). This left us with around **4.8 Million** docs of high quality of meaning (at least!).

The resulting docs are saved into a _CSV_ file to avoid extra-splitting on new line characters when saved into a text file.

## Tokenization

We didn't do much to train the tokenizer. We just used the `train_from_iterator` method from the _GPT2 pretrained tokenizer_ and passed it a generator (defined below) of all the docs to train.

```python
  def get_dataset_iterator(data, batch_size: int = 1024):
      for i in range(0, len(data), batch_size):
          yield data[i : i + batch_size]["text"]
```

The tokenizer produced number of tokens distribution for all the high quality docs that is consistent with the length distribution from above.

![Number of Tokens Distribution](./assets/num-tokens-dist-quality-docs.png)

We then saved the tokenizer to the same directory of the model (to be trained in the following step).

## GPT2-Based Models

## Performance Comparison

## Conclusions

## Credits

## License

MIT
