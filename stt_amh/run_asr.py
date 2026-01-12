#!/usr/bin/env python3
from functools import partial
import json
import re

from datasets import load_from_disk
from evaluate import load
from loguru import logger
import torch
from torchcodec.decoders import AudioDecoder
from tqdm.rich import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC
import typer
from typing_extensions import Annotated
import uroman as ur

from stt_amh.config import CV_DATASET, PRETRAINED_AMH, REPORTS_DIR

# Always enable color in loguru
logger = logger.opt(colors=True)
logger.opt = partial(logger.opt, colors=True)

app = typer.Typer()


class InferenceContext:
	def __init__(self) -> None:
		self.target_lang = "amh"
		self.model_id = "facebook/mms-1b-all"
		self.uroman = ur.Uroman()
		self.wer = load("wer")

	def load_dataset(self) -> None:
		logger.info("Loading the dataset")

		self.dataset = load_from_disk(CV_DATASET)
		self.val_ds = self.dataset["validation"]

	def load_model(self, use_custom_adapter: bool = True) -> None:
		logger.info("Loading the model")

		self.processor = AutoProcessor.from_pretrained(self.model_id, device_map="auto")
		self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, device_map="auto")

		self.processor.tokenizer.set_target_lang(self.target_lang)

		if use_custom_adapter:
			logger.info(f"Using our own <blue>{self.target_lang}</blue> adapter layers")
			self.model.load_adapter(
				self.target_lang, local_files_only=True, cache_dir=PRETRAINED_AMH, use_safetensors=True
			)
		else:
			logger.info(f"Using upstream <blue>{self.target_lang}</blue> adapter layers")
			self.model.load_adapter(self.target_lang)

	def run_asr(self, sample: AudioDecoder, reference: str) -> tuple[str, str]:
		inputs = self.processor(sample, sampling_rate=16_000, return_tensors="pt").to(self.model.device)

		with torch.no_grad():
			outputs = self.model(**inputs).logits

		ids = torch.argmax(outputs, dim=-1)[0]
		transcription = self.processor.decode(ids)
		romanized = self.uroman.romanize_string(transcription, lcode=self.target_lang)

		# Compute the WER
		wer = self.wer.compute(predictions=[transcription], references=[reference])

		# Strip tags for loguru's sake...
		safe_tr = re.sub(r"<[^<]+?>", "", transcription)
		safe_ro = re.sub(r"<[^<]+?>", "", romanized)
		logger.info(f"Transcription: <blue>{safe_tr}</blue>")
		logger.info(f"Romanized: <green>{safe_ro}</green>")
		logger.info(f"WER: <yellow>{wer:.4f}</yellow>")

		return transcription, romanized


@app.command()
def main(custom_adapter: Annotated[bool, typer.Option(help="Use our own custom adapter")] = True) -> None:
	"""Run a full inference test on the test dataset"""
	ctx = InferenceContext()
	ctx.load_dataset()
	ctx.load_model(custom_adapter)

	results = []
	for i, data_point in enumerate(tqdm(ctx.dataset["test"], desc="Inferencing...")):
		sample = data_point["audio"]["array"]
		ref = data_point["sentence"]
		transcription, romanized = ctx.run_asr(sample, ref)
		results.append(
			{
				"sample": i,
				"transcription": transcription,
				"romanized": romanized,
			}
		)

	# Dump the results to disk
	report = REPORTS_DIR / f"stt-test-{custom_adapter and 'custom' or 'stock'}.json"
	with report.open("w") as f:
		json.dump(results, f)


if __name__ == "__main__":
	app()
