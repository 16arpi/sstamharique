import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import os

    os.chdir('./mms')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pre-processing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset
    """)
    return


@app.cell
def _():
    from datasets import load_dataset, Audio

    dataset_global = load_dataset("csv", data_files="dataset.csv", split="train")
    dataset_global = dataset_global.remove_columns(["client_id", "sentence_id", "sentence_domain", "up_votes", "down_votes", "age", "gender", "accents", "variant", "locale", "segment"])
    dataset_global = dataset_global.cast_column("audio", Audio(sampling_rate=16000))
    print(dataset_global[0]["audio"])

    dataset = dataset_global.train_test_split(test_size=0.2)
    dataset_val = dataset["train"].train_test_split(test_size=0.2)
    dataset["train"] = dataset_val["train"]
    dataset["validation"] = dataset_val["test"]
    dataset
    return (dataset,)


@app.cell
def _():
    from datasets import ClassLabel, Dataset
    import random
    import polars as pl
    #from IPython.display import display, HTML

    def show_random_elements(dataset: list[str], num_examples=10):
        assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
        picks = []
        for _ in range(num_examples):
            pick = random.randint(0, len(dataset)-1)
            while pick in picks:
                pick = random.randint(0, len(dataset)-1)
            picks.append(pick)

        lf = pl.LazyFrame([dataset[i] for i in picks])
        return lf
    return Dataset, show_random_elements


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vocabulary
    """)
    return


@app.cell
def _(Dataset, dataset):
    import re

    PUNCT = re.compile(r'[!-:?፡።፣፤፥፦፧፨፠‘’“”‹›]')

    def remove_punctuation(sample: Dataset):
        # We shouldn't need to lowercase, as the alphasylabbary has no concept of caps
        sample["sentence"] = PUNCT.sub('', sample["sentence"])
        return sample

    dataset["train"] = dataset["train"].map(remove_punctuation)
    dataset["test"] = dataset["test"].map(remove_punctuation)
    dataset["validation"] = dataset["validation"].map(remove_punctuation)

    dataset["train"]
    return


@app.cell
def _(dataset, show_random_elements):
    show_random_elements(dataset["train"]["sentence"], num_examples=10)
    return


@app.cell
def _(dataset):
    # Build the full vocab set
    def extract_all_chars(batch):
      all_text = " ".join(batch["sentence"])
      vocab = list(set(all_text))
      return {"vocab": [vocab], "all_text": [all_text]}

    vocab_train = dataset["train"].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset["train"].column_names)
    vocab_test = dataset["test"].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset["test"].column_names)
    vocab_validation = dataset["validation"].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset["validation"].column_names)
    return vocab_test, vocab_train, vocab_validation


@app.cell
def _(vocab_test, vocab_train, vocab_validation):
    from rich.pretty import pprint

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]) | set(vocab_validation["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    pprint(vocab_dict)
    return (vocab_dict,)


@app.cell
def _(vocab_dict):
    target_lang = "amh"
    new_vocab_dict = {target_lang: vocab_dict}
    return new_vocab_dict, target_lang


@app.cell
def _(new_vocab_dict):
    import json
    with open('vocab.json', 'w') as vocab_file:
        json.dump(new_vocab_dict, vocab_file)
    return


@app.cell
def _(target_lang):
    from transformers import Wav2Vec2CTCTokenizer

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", target_lang=target_lang, device_map="auto")
    return (tokenizer,)


@app.cell
def _():
    #repo_name = "wav2vec2-large-mms-1b-amharique-colab"
    #tokenizer.push_to_hub(repo_name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature extractor

    A Wav2Vec2FeatureExtractor object requires the following parameters to be instantiated:

    - `feature_size`: Speech models take a sequence of feature vectors as an input. While the length of this sequence obviously varies, the feature size should not. In the case of Wav2Vec2, the feature size is 1 because the model was trained on the raw speech signal 22.
    - `sampling_rate`: The sampling rate at which the model is trained on.
    - `padding_value`: For batched inference, shorter inputs need to be padded with a specific value
    - `do_normalize`: Whether the input should be zero-mean-unit-variance normalized or not. Usually, speech models perform better when normalizing the input
    - `return_attention_mask`: Whether the model should make use of an attention_mask for batched inference. In general, XLS-R models checkpoints should always use the attention_mask.
    """)
    return


@app.cell
def _():
    from transformers import Wav2Vec2FeatureExtractor

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True, device_map="auto")
    return (feature_extractor,)


@app.cell
def _(feature_extractor, tokenizer):
    from transformers import Wav2Vec2Processor

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return


@app.cell
def _(dataset):
    dataset["train"][0]["audio"]
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
