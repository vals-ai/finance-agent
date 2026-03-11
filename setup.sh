#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y curl git build-essential

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true

# Install the finance agent package
make install

# Wrapper script so `finance-agent` is on PATH from anywhere
cat > /usr/local/bin/finance-agent << 'WRAPPER'
#!/bin/bash
source /bundle/finance-agent/.venv/bin/activate
exec finance-agent "$@"
WRAPPER
chmod +x /usr/local/bin/finance-agent

# Create logs directory
mkdir -p /logs

