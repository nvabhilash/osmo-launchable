#!/bin/bash
# Don't exit on error - we'll handle errors explicitly
set +e

# Colors
GREEN="\033[1;32m"
CYAN="\033[1;36m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

# Error handling function
error_exit() {
  echo -e "${RED}❌ Error: $1${RESET}" >&2
  exit 1
}

# Retry function
retry() {
  local max_attempts=$1
  shift
  local attempt=1
  while [ $attempt -le $max_attempts ]; do
    echo -e "${CYAN}Attempt $attempt/$max_attempts: $@${RESET}"
    if "$@"; then
      return 0
    fi
    attempt=$((attempt + 1))
    if [ $attempt -le $max_attempts ]; then
      echo -e "${YELLOW}Retrying in 5 seconds...${RESET}"
      sleep 5
    fi
  done
  return 1
}

# Exponential backoff function
exponential_backoff() {
  local attempt=$1
  local base_delay=${2:-2}  # Base delay in seconds (default 2)
  local max_delay=${3:-60}  # Max delay in seconds (default 60)
  local delay=$((base_delay * (2 ** (attempt - 1))))
  
  # Cap at max_delay
  if [ $delay -gt $max_delay ]; then
    delay=$max_delay
  fi
  
  echo $delay
}

# Wait for cluster to be ready with exponential backoff
wait_for_cluster_ready() {
  local cluster_name=$1
  local max_wait=${2:-600}  # Default 10 minutes
  local elapsed=0
  local attempt=1
  local base_delay=5
  
  echo -e "${CYAN}Waiting for cluster '${cluster_name}' to be ready (with exponential backoff)...${RESET}"
  
  while [ $elapsed -lt $max_wait ]; do
    # Check if cluster context exists first
    if ! kubectl config get-contexts kind-${cluster_name} >/dev/null 2>&1; then
      local delay=$(exponential_backoff $attempt $base_delay 60)
      echo -e "${YELLOW}Cluster context not available yet. Waiting ${delay}s before retry (attempt $attempt)...${RESET}"
      sleep $delay
      elapsed=$((elapsed + delay))
      attempt=$((attempt + 1))
      continue
    fi
    
    # Try to get nodes
    if kubectl --context=kind-${cluster_name} get nodes 2>/dev/null | grep -q "Ready"; then
      local ready_nodes=$(kubectl --context=kind-${cluster_name} get nodes --no-headers 2>/dev/null | grep -c " Ready " || echo "0")
      local total_nodes=$(kubectl --context=kind-${cluster_name} get nodes --no-headers 2>/dev/null | wc -l || echo "0")
      
      if [ "$total_nodes" -eq "0" ]; then
        local delay=$(exponential_backoff $attempt $base_delay 60)
        echo -e "${YELLOW}No nodes found yet. Waiting ${delay}s before retry (attempt $attempt)...${RESET}"
        sleep $delay
        elapsed=$((elapsed + delay))
        attempt=$((attempt + 1))
        continue
      fi
      
      if [ "$ready_nodes" -eq "$total_nodes" ] && [ "$total_nodes" -gt "0" ]; then
        echo -e "${GREEN}✓ Cluster is ready (${ready_nodes}/${total_nodes} nodes ready)${RESET}"
        return 0
      fi
      
      # Some nodes ready but not all - use shorter delay
      local delay=$(exponential_backoff $attempt 3 30)
      echo -e "${YELLOW}Cluster nodes: ${ready_nodes}/${total_nodes} ready. Waiting ${delay}s before next check (attempt $attempt)...${RESET}"
      sleep $delay
      elapsed=$((elapsed + delay))
      attempt=$((attempt + 1))
    else
      # No ready nodes yet - use exponential backoff
      local delay=$(exponential_backoff $attempt $base_delay 60)
      echo -e "${YELLOW}No ready nodes yet. Waiting ${delay}s before retry (attempt $attempt)...${RESET}"
      sleep $delay
      elapsed=$((elapsed + delay))
      attempt=$((attempt + 1))
    fi
  done
  echo -e "${RED}❌ Cluster did not become ready within ${max_wait} seconds${RESET}"
  return 1
}

echo -e "${CYAN}🛠 Checking prerequisites...${RESET}"
# Check for docker (command name, not package name)
if ! command -v docker >/dev/null 2>&1; then
  echo -e "${YELLOW}⚡ Installing docker${RESET}"
  if ! sudo apt-get update && sudo apt-get install -y docker.io; then
    error_exit "Failed to install docker"
  fi
fi

# Verify docker is running
if ! docker info >/dev/null 2>&1; then
  echo -e "${YELLOW}⚠️  Docker daemon not running. Attempting to start...${RESET}"
  sudo systemctl start docker || error_exit "Failed to start docker daemon"
  sleep 2
fi

# Check for other tools
for pkg in kind kubectl helm; do
  if ! command -v $pkg >/dev/null 2>&1; then
    echo -e "${YELLOW}⚡ Installing $pkg${RESET}"
    if ! sudo apt-get update && sudo apt-get install -y $pkg; then
      error_exit "Failed to install $pkg"
    fi
  else
    echo -e "${GREEN}✓ $pkg is installed${RESET}"
  fi
done

echo -e "${CYAN}🔍 Checking for NVIDIA GPU...${RESET}"
if ! lspci | grep -i nvidia >/dev/null 2>&1; then
  echo -e "${RED}❌ No NVIDIA GPU detected. Exiting.${RESET}"
  exit 1
fi

if ! command -v nvkind >/dev/null 2>&1; then
  echo -e "${YELLOW}⚡ Installing nvkind${RESET}"
  # Check for Go, install if needed (nvkind requires Go 1.23+)
  if ! command -v go >/dev/null 2>&1; then
    echo -e "${YELLOW}⚡ Installing Go (required for nvkind)${RESET}"
    sudo snap install go --classic
  else
    # Check if Go version is sufficient (1.23+)
    GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
    GO_MAJOR=$(echo $GO_VERSION | cut -d. -f1)
    GO_MINOR=$(echo $GO_VERSION | cut -d. -f2)
    if [ "$GO_MAJOR" -lt 1 ] || ([ "$GO_MAJOR" -eq 1 ] && [ "$GO_MINOR" -lt 23 ]); then
      echo -e "${YELLOW}⚡ Upgrading Go to version 1.23+ (required for nvkind)${RESET}"
      sudo snap install go --classic
    fi
  fi
  # Install nvkind using go install
  go install github.com/NVIDIA/nvkind/cmd/nvkind@latest
  # Ensure nvkind is in PATH (go install puts it in ~/go/bin by default)
  if [ -f "$HOME/go/bin/nvkind" ]; then
    sudo cp "$HOME/go/bin/nvkind" /usr/local/bin/nvkind
    sudo chmod +x /usr/local/bin/nvkind
  fi
fi

echo -e "${CYAN}🔑 Docker permissions...${RESET}"
sudo usermod -aG docker $USER || true

echo -e "${CYAN}🔧 Raising inotify limits...${RESET}"
echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf >/dev/null
sudo sysctl -p >/dev/null

# Ask for environment variables
read -p "Cluster name [default: osmo]: " CLUSTER_NAME
CLUSTER_NAME=${CLUSTER_NAME:-osmo}
read -p "KIND config file [default: kind-osmo-cluster-config.yaml]: " KIND_CONFIG
KIND_CONFIG=${KIND_CONFIG:-kind-osmo-cluster-config.yaml}
read -p "OSMO login user [default: testuser]: " OSMO_LOGIN_USER
OSMO_LOGIN_USER=${OSMO_LOGIN_USER:-testuser}

# Optional: NGC API key for NVIDIA Helm registry (for OSMO charts)
echo -e "${CYAN}If you need to access NVIDIA OSMO charts, generate an NGC API key at:${RESET} https://ngc.nvidia.com/setup/api-key"
echo -e "${CYAN}Enter your NGC API Key (leave blank to skip):${RESET}"
read -s NGC_API_KEY || true
echo
export NGC_API_KEY

echo -e "${CYAN}📝 Creating KIND config...${RESET}"
# Ensure mount directories exist
mkdir -p /tmp/localstack-s3 || echo -e "${YELLOW}⚠️  Could not create /tmp/localstack-s3 (may need sudo)${RESET}"

# Create full cluster configuration with all required nodes
cat > $KIND_CONFIG <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: $CLUSTER_NAME
nodes:
- role: control-plane
- role: worker
  extraPortMappings:
  - containerPort: 30080
    hostPort: 80
    protocol: TCP
  kubeadmConfigPatches:
  - |
    kind: JoinConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "node_group=ingress,nvidia.com/gpu.deploy.operands=false"
- role: worker
  kubeadmConfigPatches:
  - |
    kind: JoinConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "node_group=kai-scheduler,nvidia.com/gpu.deploy.operands=false"
- role: worker
  extraMounts:
  - hostPath: /dev/null
    containerPath: /var/run/nvidia-container-devices/all
  kubeadmConfigPatches:
  - |
    kind: JoinConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "node_group=compute"
EOF

echo -e "${CYAN}🚀 Creating KIND cluster...${RESET}"
# Delete existing cluster if it exists
if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
  echo -e "${YELLOW}⚠️  Cluster '${CLUSTER_NAME}' already exists. Deleting it first...${RESET}"
  kind delete cluster --name=${CLUSTER_NAME} 2>/dev/null || nvkind cluster delete --name=${CLUSTER_NAME} 2>/dev/null || true
  # Wait a bit for cleanup
  sleep 5
fi

# Create cluster with retries
CLUSTER_CREATED=false
for attempt in 1 2 3; do
  echo -e "${CYAN}Creating cluster (attempt $attempt/3)...${RESET}"
  
  # Try with nvkind first
  # Start cluster creation and use exponential backoff for health checks
  echo -e "${CYAN}Initiating cluster creation...${RESET}"
  nvkind cluster create --config-template=$KIND_CONFIG >/tmp/nvkind-create.log 2>&1 &
  create_pid=$!
  
  # Initial wait before first health check (exponential backoff starts here)
  initial_wait=15
  echo -e "${CYAN}Waiting ${initial_wait}s before first health check (exponential backoff will follow)...${RESET}"
  sleep $initial_wait
  
  # Check if cluster was created (with exponential backoff)
  check_attempt=1
  max_checks=15
  while [ $check_attempt -le $max_checks ]; do
    # Check if process completed
    if ! kill -0 $create_pid 2>/dev/null; then
      wait $create_pid
      create_exit=$?
      if [ $create_exit -eq 0 ]; then
        echo -e "${GREEN}✓ Cluster creation process completed successfully${RESET}"
        CLUSTER_CREATED=true
        break
      else
        echo -e "${YELLOW}Cluster creation process exited with error. Checking if cluster exists anyway...${RESET}"
        # Still check if cluster was created despite error
      fi
    fi
    
    # Check if cluster exists
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
      echo -e "${GREEN}✓ Cluster '${CLUSTER_NAME}' detected!${RESET}"
      # Wait for process to complete if still running
      if kill -0 $create_pid 2>/dev/null; then
        echo -e "${CYAN}Waiting for cluster creation process to finish...${RESET}"
        wait $create_pid 2>/dev/null || true
      fi
      CLUSTER_CREATED=true
      break
    fi
    
    # Exponential backoff between health checks
    delay=$(exponential_backoff $check_attempt 5 30)
    echo -e "${CYAN}Cluster not found yet. Waiting ${delay}s before next check (attempt $check_attempt/$max_checks)...${RESET}"
    sleep $delay
    check_attempt=$((check_attempt + 1))
  done
  
  # If we found the cluster, break out of retry loop
  if [ "$CLUSTER_CREATED" = true ]; then
    break
  fi
  
  # If process is still running, terminate it
  if kill -0 $create_pid 2>/dev/null; then
    echo -e "${YELLOW}Cluster creation process still running but cluster not found after ${max_checks} checks. Terminating...${RESET}"
    kill $create_pid 2>/dev/null || true
    wait $create_pid 2>/dev/null || true
  fi
  
  # If nvkind fails, try with regular kind as fallback (simpler, more reliable)
  if [ $attempt -eq 2 ]; then
    echo -e "${YELLOW}⚠️  nvkind failed. Trying with regular kind (simpler configuration)...${RESET}"
    # Create a minimal config for kind
    cat > /tmp/kind-minimal-config.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: $CLUSTER_NAME
nodes:
- role: control-plane
- role: worker
- role: worker
- role: worker
EOF
    if kind create cluster --config=/tmp/kind-minimal-config.yaml 2>&1 | tee /tmp/kind-create.log; then
      CLUSTER_CREATED=true
      echo -e "${YELLOW}⚠️  Created cluster with regular kind (minimal config). Some features may be limited.${RESET}"
      break
    fi
  fi
  
  if [ $attempt -lt 3 ]; then
    echo -e "${YELLOW}Cluster creation failed. Cleaning up and retrying...${RESET}"
    kind delete cluster --name=${CLUSTER_NAME} 2>/dev/null || true
    sleep 15
  fi
done

if [ "$CLUSTER_CREATED" = false ]; then
  error_exit "Failed to create cluster after 3 attempts. Check /tmp/nvkind-create.log for details."
fi

# Wait for cluster to be ready
if ! wait_for_cluster_ready "${CLUSTER_NAME}" 600; then
  echo -e "${YELLOW}⚠️  Cluster may not be fully ready, but continuing...${RESET}"
  kubectl --context=kind-${CLUSTER_NAME} get nodes || true
fi

# Set kubectl context
kubectl config use-context kind-${CLUSTER_NAME} || error_exit "Failed to set kubectl context"

echo -e "${CYAN}▶ Installing GPU Operator...${RESET}"
if [ ! -f "gpu-operator-v25.10.0.tgz" ]; then
  echo -e "${CYAN}Downloading GPU Operator chart...${RESET}"
  if ! retry 3 helm fetch https://helm.ngc.nvidia.com/nvidia/charts/gpu-operator-v25.10.0.tgz; then
    error_exit "Failed to download GPU Operator chart"
  fi
fi

if ! retry 3 helm upgrade --install gpu-operator gpu-operator-v25.10.0.tgz \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \
  --set toolkit.enabled=false \
  --set nfd.enabled=true \
  --wait --timeout=10m; then
  echo -e "${YELLOW}⚠️  GPU Operator installation had issues. Continuing...${RESET}"
  kubectl get pods -n gpu-operator || true
fi

echo -e "${CYAN}▶ Installing KAI scheduler...${RESET}"
if ! retry 3 helm upgrade --install kai-scheduler \
  oci://ghcr.io/nvidia/kai-scheduler/kai-scheduler \
  --version v0.8.1 \
  --create-namespace -n kai-scheduler \
  --set global.nodeSelector.node_group=kai-scheduler \
  --set scheduler.additionalArgs[0]=--default-staleness-grace-period=-1s \
  --set scheduler.additionalArgs[1]=--update-pod-eviction-condition=true \
  --wait --timeout=10m; then
  echo -e "${YELLOW}⚠️  KAI scheduler installation had issues. Continuing...${RESET}"
  kubectl get pods -n kai-scheduler || true
fi

echo -e "${CYAN}▶ Installing OSMO...${RESET}"
if [ ! -f "quick-start-1.0.0.tgz" ]; then
  echo -e "${CYAN}Downloading OSMO chart...${RESET}"
  if [ -n "$NGC_API_KEY" ]; then
    # Authenticated pull via NGC repo if API key provided
    helm repo add nvidian https://helm.ngc.nvidia.com/nvidian/charts --username '$oauthtoken' --password "$NGC_API_KEY" >/dev/null 2>&1 || true
    helm repo update >/dev/null 2>&1 || true
    if ! retry 3 helm pull nvidian/quick-start --version 1.0.0; then
      error_exit "Failed to download OSMO chart from authenticated NGC repo"
    fi
  else
    # Fallback to direct URL (may require auth and fail with 401)
    if ! retry 3 helm fetch https://helm.ngc.nvidia.com/nvidian/osmo/charts/quick-start-1.0.0.tgz; then
      error_exit "Failed to download OSMO chart"
    fi
  fi
fi

if ! retry 3 helm upgrade --install osmo quick-start-1.0.0.tgz --namespace osmo --create-namespace --wait --timeout=15m; then
  echo -e "${YELLOW}⚠️  OSMO installation had issues. Continuing...${RESET}"
  kubectl get pods -n osmo || true
fi

echo -e "${CYAN}🔗 Adding host entry...${RESET}"
if ! grep -q "quick-start.osmo" /etc/hosts 2>/dev/null; then
  echo "127.0.0.1 quick-start.osmo" | sudo tee -a /etc/hosts >/dev/null || echo -e "${YELLOW}⚠️  Failed to add host entry (may need manual addition)${RESET}"
else
  echo -e "${GREEN}✓ Host entry already exists${RESET}"
fi

echo -e "${CYAN}⬇ Installing OSMO CLI...${RESET}"
if ! command -v osmo >/dev/null 2>&1; then
  if ! retry 3 bash -c 'curl -fsSL https://raw.githubusercontent.com/NVIDIA/OSMO/refs/heads/main/install.sh | bash'; then
    echo -e "${YELLOW}⚠️  Failed to install OSMO CLI. You may need to install it manually.${RESET}"
  fi
else
  echo -e "${GREEN}✓ OSMO CLI is already installed${RESET}"
fi

echo -e "${GREEN}✅ Setup complete!${RESET}"
echo -e "${GREEN}✅ Login with:${RESET} osmo login http://quick-start.osmo --method=dev --username=$OSMO_LOGIN_USER"
echo -e "${GREEN}✅ Check pods:${RESET} kubectl get pods --namespace osmo"
echo -e "${GREEN}✅ Check cluster:${RESET} kubectl get nodes"