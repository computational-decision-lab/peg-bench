.PHONY: demo preprocess train-vqvae train-baseline build-preferences train-dpo evaluate test

demo:
	python scripts/run_demo.py

preprocess:
	python scripts/preprocess.py --config configs/data.yaml

train-vqvae:
	python scripts/train_vqvae.py --config configs/vqvae.yaml

train-baseline:
	python scripts/train_baseline.py --config configs/baseline.yaml

build-preferences:
	python scripts/build_preferences.py --encoded outputs/vqvae/encoded_tokens.npz --dataset outputs/data/volume_bars.npz --checkpoint outputs/baseline/best_baseline.pt --output outputs/preferences/preferences.pkl

train-dpo:
	python scripts/train_dpo.py --preferences outputs/preferences/preferences.pkl --reference-checkpoint outputs/baseline/best_baseline.pt --output-dir outputs/dpo

evaluate:
	python scripts/evaluate_all.py --backtest-config configs/backtest.yaml

test:
	python -m unittest discover -s tests -v
