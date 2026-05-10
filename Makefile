.PHONY: install install-dev install-train lint format test test-slow embed recommend export quantize verify bench eval download-fma download-mtat download-clap-music distill-cache distill-train student-benchmark student-embed student-export student-quantize clean

PYTHON ?= python
ONNX_FP32 ?= models/onnx/mert_v1_95m_5s.onnx
ONNX_INT8 ?= models/onnx/mert_v1_95m_5s.int8.onnx
CLAP_MUSIC_CKPT ?= models/clap/music_audioset_epoch_15_esc_90.14.pt
FMA_TARGETS ?= models/student/fma_small_clap_targets.parquet
STUDENT_CKPT ?= models/student/mel_cnn_student.pt
STUDENT_ONNX_FP32 ?= models/onnx/mel_cnn_student.onnx
STUDENT_ONNX_INT8 ?= models/onnx/mel_cnn_student.int8.onnx
WANDB_PROJECT ?=

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-recommend:
	$(PYTHON) -m pip install -e ".[dev,recommend]"

install-train:
	$(PYTHON) -m pip install -e ".[dev,train]"

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

test-slow:
	$(PYTHON) -m pytest -m slow

embed:
	@if [ -z "$(ROOT)" ] || [ -z "$(OUT)" ]; then \
		echo "Usage: make embed ROOT=<audio-dir> OUT=<store.parquet>"; exit 2; \
	fi
	$(PYTHON) -m predictor.cli embed --root "$(ROOT)" --out "$(OUT)"

recommend:
	@if [ -z "$(STORE)" ] || [ -z "$(SEED)" ]; then \
		echo "Usage: make recommend STORE=<store.parquet> SEED=<track_id> [K=10]"; exit 2; \
	fi
	$(PYTHON) -m predictor.cli recommend --store "$(STORE)" --seed-id "$(SEED)" -k $${K:-10}

export:
	$(PYTHON) -m conversion.export_onnx --out "$(ONNX_FP32)"

quantize:
	$(PYTHON) -m conversion.quantize --in "$(ONNX_FP32)" --out "$(ONNX_INT8)"

verify:
	$(PYTHON) -m conversion.verify --onnx "$(ONNX_INT8)" --n $${N:-50}

bench:
	@if [ -z "$(ONNX)" ]; then \
		echo "Usage: make bench ONNX=<path.onnx>"; exit 2; \
	fi
	$(PYTHON) -m conversion.bench --onnx "$(ONNX)"

eval:
	@if [ -z "$(STORE)" ] || [ -z "$(DATASET)" ]; then \
		echo "Usage: make eval STORE=<store.parquet> DATASET=fma_small|mtat"; exit 2; \
	fi
	$(PYTHON) -m eval.run_eval --store "$(STORE)" --dataset "$(DATASET)"

download-fma:
	$(PYTHON) scripts/download_fma.py $${SUBSET:+--subset $$SUBSET} $${KEEP_ZIP:+--keep-zip}

download-mtat:
	$(PYTHON) scripts/download_mtat.py

download-clap-music:
	$(PYTHON) scripts/download_clap_music.py --out "$(CLAP_MUSIC_CKPT)"

distill-cache:
	$(PYTHON) -m student.prepare_clap_targets \
		--teacher-checkpoint "$(CLAP_MUSIC_CKPT)" \
		--out "$(FMA_TARGETS)" \
		--windows-per-track $${WINDOWS:-1} \
		--batch-size $${BATCH:-8} \
		$${LIMIT:+--limit $$LIMIT} \
		$${WANDB_PROJECT:+--wandb-project $$WANDB_PROJECT} \
		--wandb-tag dataset:fma-small --wandb-tag teacher:laion-clap-music

panns-cache:
	$(PYTHON) scripts/prepare_panns_targets.py \
		--audio-root "$${AUDIO_ROOT:-data/raw/fma_medium}" \
		--metadata-root "$${METADATA_ROOT:-data/raw/fma_metadata}" \
		--subset $${SUBSET:-medium} \
		--out "$${PANNS_TARGETS:-models/student/fma_$${SUBSET:-medium}_panns_targets.parquet}" \
		--device $${DEVICE:-auto} \
		--batch-size $${BATCH:-64} \
		--num-workers $${WORKERS:-8} \
		--windows-per-track $${WINDOWS:-1} \
		$${LIMIT:+--limit $$LIMIT} \
		$${WANDB_PROJECT:+--wandb-project $$WANDB_PROJECT} \
		--wandb-tag teacher:panns-cnn14

distill-train:
	$(PYTHON) -m student.train_distill \
		--targets "$(FMA_TARGETS)" \
		--out "$(STUDENT_CKPT)" \
		--epochs $${EPOCHS:-5} \
		--batch-size $${BATCH:-32} \
		$${DEVICE:+--device $$DEVICE} \
		$${LIMIT_TRAIN:+--limit-train $$LIMIT_TRAIN} \
		$${LIMIT_VAL:+--limit-val $$LIMIT_VAL} \
		$${WANDB_PROJECT:+--wandb-project $$WANDB_PROJECT} \
		--wandb-tag dataset:fma-small --wandb-tag student:mel-cnn

student-benchmark:
	$(PYTHON) -m student.benchmark \
		--targets "$(FMA_TARGETS)" \
		--checkpoint "$(STUDENT_CKPT)" \
		--split $${SPLIT:-test} \
		--batch-size $${BATCH:-32} \
		$${DEVICE:+--device $$DEVICE} \
		$${LIMIT:+--limit $$LIMIT} \
		$${TEACHER_LATENCY:+--teacher-checkpoint "$(CLAP_MUSIC_CKPT)"} \
		$${WANDB_PROJECT:+--wandb-project $$WANDB_PROJECT} \
		--wandb-tag dataset:fma-small --wandb-tag compare:laion-clap

student-embed:
	@if [ -z "$(OUT)" ]; then \
		echo "Usage: make student-embed OUT=<store.parquet> [SPLIT=test]"; exit 2; \
	fi
	$(PYTHON) -m student.embed \
		--targets "$(FMA_TARGETS)" \
		--checkpoint "$(STUDENT_CKPT)" \
		--out "$(OUT)" \
		$${SPLIT:+--split $$SPLIT}

student-export:
	$(PYTHON) -m student.export_onnx --checkpoint "$(STUDENT_CKPT)" --out "$(STUDENT_ONNX_FP32)"

student-quantize:
	$(PYTHON) -m conversion.quantize --in "$(STUDENT_ONNX_FP32)" --out "$(STUDENT_ONNX_INT8)"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
