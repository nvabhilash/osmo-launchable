#!/bin/bash

set -e

# 1. Prerequisites: Ensure Docker, KIND, kubectl, osmo, curl, helm are installed

# 2. Detect architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    ARCH="amd64"
elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    ARCH="arm64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi
echo "Detected architecture: $ARCH"

# 3. Set environment variables (fill in your values)
export CONTAINER_REGISTRY_PASSWORD="<your-ngc-api-key>"
export S3_ENDPOINT="<your-s3-endpoint>"
export S3_ACCESS_KEY_ID="<your-access-key-id>"
export S3_ACCESS_KEY="<your-access-key>"
export S3_REGION="<your-s3-region>"
export GITLAB_TOKEN="<your-gitlab-token>"

# 4. Download KIND cluster config
echo "PRIVATE-TOKEN: $GITLAB_TOKEN" > /tmp/gitlab_header
curl -H "@/tmp/gitlab_header" \
    -o kind-osmo-cluster-config.yaml \
    "https://gitlab-master.nvidia.com/api/v4/projects/72729/repository/files/build%2Fkind-osmo-cluster-config.yaml/raw?ref=main"

# 5. Create KIND cluster
kind create cluster --config kind-osmo-cluster-config.yaml --name osmo

# 6. Add Helm repo
helm repo add nvstaging-osmo https://helm.ngc.nvidia.com/nvstaging/osmo \
    --username='$oauthtoken' --password="$CONTAINER_REGISTRY_PASSWORD"
helm repo update

# 7. Install Helm chart
helm upgrade --install osmo nvstaging-osmo/osmo-quick-start \
    --version 1.0.0-latest \
    --namespace osmo \
    --create-namespace \
    --set global.containerRegistry.password="$CONTAINER_REGISTRY_PASSWORD" \
    --set global.objectStorage.endpoint="$S3_ENDPOINT" \
    --set global.objectStorage.accessKeyId="$S3_ACCESS_KEY_ID" \
    --set global.objectStorage.accessKey="$S3_ACCESS_KEY" \
    --set global.objectStorage.region="$S3_REGION" \
    --set global.nodeSelector."kubernetes.io/arch"=$ARCH \
    --set ingress-nginx.controller.nodeSelector."kubernetes.io/arch"=$ARCH

# 8. Update /etc/hosts
sudo bash -c 'echo "127.0.0.1 osmo-ingress-nginx-controller.osmo.svc.cluster.local" >> /etc/hosts'

# 9. Login to OSMO
osmo login http://osmo-ingress-nginx-controller.osmo.svc.cluster.local --method=dev --username=testuser

# 10. Download hello world workflow file
curl -H "@/tmp/gitlab_header" \
    -o hello_world.yaml \
    "https://gitlab-master.nvidia.com/api/v4/projects/72729/repository/files/external%2Fdocs%2Fsamples%2Fhello_world%2Fhello_world.yaml/raw?ref=main"

# 11. Submit workflow
osmo workflow submit hello_world.yaml