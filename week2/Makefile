.PHONY: reproduce

reproduce:
	cargo build --release --manifest-path md/Cargo.toml
	./md/target/release/md --n 100 --rho 0.8 --temperature 0.5 --dt 0.01 --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026 --out artifacts > /dev/null
	python3 scripts/check_evidence.py
