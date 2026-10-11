# Every target is a plain python command; the Makefile only saves typing.
PYTHON ?= python

.PHONY: help install test quick headline figures sweeps bases report all clean

help:
	@echo "make install    install the requirements"
	@echo "make test       run the test suite"
	@echo "make quick      train the MLP baseline and draw its figures (about 20 minutes)"
	@echo "make headline   train the transformer run the README shows (about 2 hours on 8 CPU threads)"
	@echo "make figures    redraw the figures from the finished headline run"
	@echo "make sweeps     sweep training fraction, weight decay and seed"
	@echo "make bases      train the add/div pair and draw the two basis comparison"
	@echo "make report     rebuild RESULTS.md from every run"
	@echo "make all        headline, bases, sweeps, figures and report"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

quick:
	$(PYTHON) -m grokking.train --config configs/quick_mlp.json
	$(PYTHON) -m grokking.figures --run runs/quick_mlp --out figures/quick_mlp

headline:
	$(PYTHON) -m grokking.train --config configs/headline.json

figures:
	$(PYTHON) -m grokking.figures --run runs/headline --out figures
	$(PYTHON) -m grokking.figures --sweep runs/sweeps/sweep_train_frac.json --out figures
	$(PYTHON) -m grokking.figures --sweep runs/sweeps/sweep_weight_decay.json --out figures
	$(PYTHON) -m grokking.figures --seeds runs/sweeps --out figures
	$(PYTHON) -m grokking.figures --two-bases runs/add_p59 runs/div_p59 --out figures

sweeps:
	$(PYTHON) -m grokking.sweep --config configs/sweep_mlp.json --vary train_frac \
		--values 0.2 0.3 0.4 0.5 0.6 --out runs/sweeps --workers 2 --threads-per-worker 2
	$(PYTHON) -m grokking.sweep --config configs/sweep_mlp.json --vary weight_decay \
		--values 0 0.25 0.5 1.0 2.0 4.0 --out runs/sweeps --workers 2 --threads-per-worker 2
	$(PYTHON) -m grokking.sweep --config configs/sweep_mlp.json --vary seed \
		--values 0 1 2 3 4 5 --out runs/sweeps --workers 2 --threads-per-worker 2

bases:
	$(PYTHON) -m grokking.train --config configs/add_p59.json
	$(PYTHON) -m grokking.train --config configs/div_p59.json
	$(PYTHON) -m grokking.figures --two-bases runs/add_p59 runs/div_p59 --out figures

report:
	$(PYTHON) scripts/report.py --runs runs --out RESULTS.md

all: headline bases sweeps figures report

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache logs_*.txt
