# BIOS Career Check — common tasks
.PHONY: test eval train dataset ext check-shared

SHARED_SRC    = shared/biosClient.js
SHARED_COPIES = frontend/src/shared/biosClient.js extension/shared/biosClient.js

ext:             ## sync the shared client module into web app + extension
	@for f in $(SHARED_COPIES); do \
	  mkdir -p $$(dirname $$f); \
	  { echo "// AUTO-SYNCED from /$(SHARED_SRC) — edit that file, then run: make ext"; \
	    cat $(SHARED_SRC); } > $$f; \
	  echo "synced $$f"; \
	done

check-shared:    ## fail if the synced copies drifted from /shared
	@for f in $(SHARED_COPIES); do \
	  tail -n +2 $$f | diff -q - $(SHARED_SRC) >/dev/null \
	    || { echo "DRIFT: $$f != $(SHARED_SRC) — run make ext"; exit 1; }; \
	done; echo "shared copies in sync"

test:            ## calibration acceptance + contract + regression tests
	python3 tests/test_calibration.py

eval:            ## run the Phase 2 evaluation harness (writes eval/report.md + eval/metrics.json)
	python3 eval/run_eval.py

dataset:         ## rebuild the labeled eval dataset
	python3 eval/build_dataset.py

train:           ## retrain the Layer 2 classifier (needs ml/requirements-train.txt)
	python3 ml/train_layer2.py
