#!/usr/bin/env python3
#
##

from contextlib import chdir
import re
from rich.pretty import pprint
from typing import Any

from datasets import load_dataset, Audio, Dataset, DatasetDict

def import_dataset() -> DatasetDict:
	with chdir("mms"):
		# There aren't any actual splits yet, so everything's in the default train
		dataset_full = load_dataset("csv", data_files="dataset.csv", split="train")

	dataset_full = dataset_full.remove_columns(["client_id", "sentence_id", "sentence_domain", "up_votes", "down_votes", "age", "gender", "accents", "variant", "locale", "segment"])
	dataset_full = dataset_full.cast_column("audio", Audio(sampling_rate=16000))

	dataset = dataset_full.train_test_split(test_size=0.2, seed=42)
	dataset_val = dataset["train"].train_test_split(test_size=0.2, seed=42)
	dataset["train"] = dataset_val["train"]
	dataset["validation"] = dataset_val["test"]

	return dataset

def preprocess_dataset(dataset: DatasetDict) -> DatasetDict:
	PUNCT = re.compile(r'[!-:?፡።፣፤፥፦፧፨፠‘’“”‹›]')

	def remove_punctuation(example: dict[str, Any]) -> dict[str, Any]:
		# We shouldn't need to lowercase, as the alphasyllabary has no concept of caps
		example["sentence"] = PUNCT.sub('', example["sentence"])
		return example

	dataset["train"] = dataset["train"].map(remove_punctuation)
	dataset["test"] = dataset["test"].map(remove_punctuation)
	dataset["validation"] = dataset["validation"].map(remove_punctuation)

	return dataset

def main() -> None:
	ds = import_dataset()
	ds = preprocess_dataset(ds)

	pprint(ds)
	with chdir("mms"):
		ds.save_to_disk("cv_amh_data")

if __name__ == "__main__":
	main()
