# ffmpeg-mcp — common tasks. Run `make` or `make help` for the list.
#
# Destructive targets ask before deleting. Pass CONFIRM=yes to skip the prompt
# in a script.

SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= uv
NPM ?= npm
UI_SRC := ui-src
UI_DIST := src/ffmpeg_mcp/ui/static

UI_HOST ?= 127.0.0.1
UI_PORT ?= 8756
VITE_PORT ?= 5173

# The workspace holds the job store, rendered outputs, uploads, downloaded
# ffmpeg builds and models. This mirrors the server's own default; if you have
# FFMPEG_MCP_WORKSPACE set, that wins, so the clean targets always point at the
# same place the server actually writes to.
WORKSPACE ?= $(if $(FFMPEG_MCP_WORKSPACE),$(FFMPEG_MCP_WORKSPACE),$(HOME)/.ffmpeg-mcp/workspace)

define confirm
	if [ "$(CONFIRM)" != "yes" ]; then \
	  read -r -p "$(1) Continue? [y/N] " reply; \
	  case "$$reply" in y|Y|yes|YES) ;; *) echo "Aborted."; exit 1 ;; esac; \
	fi
endef

.PHONY: help setup install ui-install server ui dev ui-build \
        test test-unit test-integration lint format typecheck check \
        info mcp-config clean clean-data reset

# --------------------------------------------------------------------------- #
# Help
# --------------------------------------------------------------------------- #

help: ## Show this list
	@echo "ffmpeg-mcp"
	@echo
	@grep -hE '^[a-z][a-z-]*:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  workspace: $(WORKSPACE)"

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

setup: install ## Install everything needed to run the server and the UI

install: ## Install Python deps (core + whisper + vision + ui + dev tools)
	$(UV) sync --all-extras

ui-install: ## Install the Node toolchain (only needed to rebuild the frontend)
	cd $(UI_SRC) && $(NPM) install

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

server: ## Run the MCP server on stdio (this is what an MCP client launches)
	$(UV) run ffmpeg-mcp-server

ui: ## Run the local editor UI (http://127.0.0.1:8756 by default)
	$(UV) run ffmpeg-mcp-ui --host $(UI_HOST) --port $(UI_PORT)

dev: ## Run the UI backend plus the Vite dev server with hot reload
	@echo "backend  http://$(UI_HOST):$(UI_PORT)"
	@echo "frontend http://localhost:$(VITE_PORT)   <- open this one"
	@trap 'kill 0' EXIT INT TERM; \
	 $(UV) run ffmpeg-mcp-ui --host $(UI_HOST) --port $(UI_PORT) & \
	 cd $(UI_SRC) && $(NPM) run dev

ui-build: ## Rebuild the frontend into the Python package
	cd $(UI_SRC) && $(NPM) run build

# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #

test: ## Run the full test suite
	$(UV) run pytest

test-unit: ## Run only the tests that need no ffmpeg binary
	$(UV) run pytest -m "not integration"

test-integration: ## Run only the tests that render real media
	$(UV) run pytest -m integration

lint: ## Check lint and formatting
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Fix lint and formatting in place
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## Run mypy
	$(UV) run mypy

check: lint typecheck test ## Lint, typecheck and test — everything CI would run

# --------------------------------------------------------------------------- #
# Inspect
# --------------------------------------------------------------------------- #

info: ## Show the resolved workspace, ffmpeg binary and tool count
	@$(UV) run python -c "$$INFO_SCRIPT"

mcp-config: ## Print the MCP client config block with absolute paths filled in
	@printf '{\n  "mcpServers": {\n    "ffmpeg-mcp": {\n      "command": "%s",\n      "args": ["run", "--directory", "%s", "ffmpeg-mcp-server"]\n    }\n  }\n}\n' \
	  "$$(command -v $(UV))" "$(CURDIR)"

define INFO_SCRIPT
from ffmpeg_mcp.config import get_settings
from ffmpeg_mcp.tools.registry import load_all_tools
s = get_settings()
print(f"workspace     {s.workspace}")
print(f"allowed roots {', '.join(str(r) for r in s.allowed_roots)}")
print(f"whisper model {s.whisper_model}")
print(f"tools         {len(load_all_tools())}")
try:
    from ffmpeg_mcp.binaries import get_binaries
    b = get_binaries(s)
    print(f"ffmpeg        {b.ffmpeg} ({b.source}, {' '.join(b.version.split()[:3])})")
except Exception as exc:
    print(f"ffmpeg        unresolved ({exc})")
endef
export INFO_SCRIPT

# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #

clean: ## Remove build and cache artefacts from the repo (your media is untouched)
	rm -rf .pytest_cache .ruff_cache .mypy_cache tests/fixtures
	find . -path ./.venv -prune -o -name __pycache__ -type d -exec rm -rf {} +
	rm -f $(UI_SRC)/tsconfig.tsbuildinfo

clean-data: ## Delete jobs, rendered outputs and uploads (keeps downloaded ffmpeg and models)
	@test -n "$(WORKSPACE)" || { echo "WORKSPACE is empty; refusing to delete."; exit 1; }
	@$(call confirm,This deletes every job and rendered output under $(WORKSPACE).)
	rm -rf "$(WORKSPACE)/jobs" "$(WORKSPACE)/uploads" "$(WORKSPACE)/projects" "$(WORKSPACE)/jobs.db"
	@echo "Cleared. Downloaded binaries and models kept."

reset: clean ## Wipe the whole workspace, including downloaded ffmpeg and models
	@test -n "$(WORKSPACE)" || { echo "WORKSPACE is empty; refusing to delete."; exit 1; }
	@$(call confirm,This deletes ALL of $(WORKSPACE), including downloaded ffmpeg builds and models.)
	rm -rf "$(WORKSPACE)"
	@echo "Workspace removed. It is recreated on the next run."
