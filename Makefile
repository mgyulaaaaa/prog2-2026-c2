.PHONY: run-all comp-table run clean-logs

run-all:
	uv run python collective_run.py

comp-table:
	uv run python comptable.py

run:
	uv run python single_run.py $(SOLUTION) $(if $(COMPARE),--compare $(COMPARE),)

clean-logs:
	git clean -f runs/run-logs
