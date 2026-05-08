.PHONY: install install-dev lint format test test-slow embed recommend export quantize verify bench eval download-fma download-mtat clean

PYTHON ?= python
ONNX_FP32 ?= models/onnx/mert_v1_95m_5s.onnx
ONNX_INT8 ?= models/onnx/mert_v1_95m_5s.int8.onnx

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-recommend:
	$(PYTHON) -m pip install -e ".[dev,recommend]"

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
	$(PYTHON) scripts/download_fma.py

download-mtat:
	$(PYTHON) scripts/download_mtat.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
