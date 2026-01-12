#!/usr/bin/env python3
#
##

from datasets import load_from_disk, DatasetDict
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

def load_val_dataset() -> DatasetDict:
	ds = load_from_disk("mms/cv_amh_data")
	return ds["validation"]

def load_model():
	model_id = "facebook/mms-1b-all"

	processor = AutoProcessor.from_pretrained(model_id)
	model = Wav2Vec2ForCTC.from_pretrained(model_id)

	processor.tokenizer.set_target_lang("amh")
	model.load_adapter("wav2vec2-large-mms-1b-amharic-cv/amh", local_files_only=True)

	return processor, model

def run_asr(processor, model, sample) -> str:
	inputs = processor(sample, sampling_rate=16_000, return_tensors="pt")

	with torch.no_grad():
		outputs = model(**inputs).logits

	ids = torch.argmax(outputs, dim=-1)[0]
	transcription = processor.decode(ids)

def main() -> None:
	val_data = load_val_dataset()
	amh_sample = next(iter(val_data))["audio"]["array"]

	processor, model = load_model()
	transcription = run_asr(processor, model, sample)

if __name__ == "__main__":
	main()
