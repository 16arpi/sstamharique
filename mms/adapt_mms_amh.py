import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import os
    import platform

    # Try not to blow past 24GB of VRAM...
    # Spoiler alert: it does ;'(.
    if platform.node() == "ahsoka":
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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
    return Dataset, random, show_random_elements


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
    with open("vocab.json", "w") as vocab_file:
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
    return Wav2Vec2Processor, processor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preprocess Data
    """)
    return


@app.cell
def _(dataset):
    dataset["train"][0]["audio"]
    return


@app.cell
def _(dataset, random):
    rand_int = random.randint(0, len(dataset["train"])-1)

    print("Target text:", dataset["train"][rand_int]["sentence"])
    print("Input array shape:", dataset["train"][rand_int]["audio"]["array"].shape)
    print("Sampling rate:", dataset["train"][rand_int]["audio"]["sampling_rate"])
    return


@app.cell
def _(processor):
    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch
    return (prepare_dataset,)


@app.cell
def _(dataset, prepare_dataset):
    dataset["train"] = dataset["train"].map(prepare_dataset, remove_columns=dataset["train"].column_names)
    dataset["test"] = dataset["test"].map(prepare_dataset, remove_columns=dataset["test"].column_names)
    dataset["validation"] = dataset["validation"].map(prepare_dataset, remove_columns=dataset["validation"].column_names)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training
    """)
    return


@app.cell
def _(Wav2Vec2Processor):
    import torch

    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
        """

        processor: Wav2Vec2Processor
        padding: bool | str = True

        def __call__(self, features: list[dict[str, list[int] | torch.Tensor]]) -> dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )

            labels_batch = self.processor.pad(
                labels=label_features,
                padding=self.padding,
                return_tensors="pt",
            )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch
    return (DataCollatorCTCWithPadding,)


@app.cell
def _(DataCollatorCTCWithPadding, processor):
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    return (data_collator,)


@app.cell
def _():
    from evaluate import load

    wer_metric = load("wer")
    return (wer_metric,)


@app.cell
def _(np, processor, wer_metric):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    return (compute_metrics,)


@app.cell
def _(processor):
    from transformers import Wav2Vec2ForCTC

    # TODO: Tweak hyperparameters
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/mms-1b-all",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
        device_map="auto",
    )
    return (model,)


@app.cell
def _(model):
    model.init_adapter_layers()
    return


@app.cell
def _(model):
    # Freeze all weights, but the adapter layers
    model.freeze_base_model()

    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True
    return


@app.cell
def _():
    from transformers import TrainingArguments

    training_args = TrainingArguments(
      output_dir="wav2vec2-large-mms-1b-amharic-cv",
      group_by_length=True,
      per_device_train_batch_size=16, # Lowered from 32 to lower memory pressure
      gradient_accumulation_steps=1, # TODO: We might need to increase this to lower memory pressure some more
      eval_strategy="steps",
      num_train_epochs=4,
      gradient_checkpointing=True,
      fp16=True,
      save_steps=200,
      eval_steps=100,
      logging_steps=100,
      learning_rate=1e-3,
      warmup_steps=100,
      save_total_limit=2,
      push_to_hub=False,
    )
    return (training_args,)


@app.cell
def _(
    compute_metrics,
    data_collator,
    dataset,
    model,
    processor,
    training_args,
):
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=processor.feature_extractor,
    )
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell
def _(model, target_lang, training_args):
    from safetensors.torch import save_file as safe_save_file
    from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE

    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
    adapter_file = os.path.join(training_args.output_dir, adapter_file)

    safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
