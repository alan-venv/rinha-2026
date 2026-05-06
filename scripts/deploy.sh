#!/bin/bash

VERSION=`date +%s`
TAG="ghcr.io/alan-venv/rinha-service-2026:${VERSION}"

docker compose -f docker/docker-compose.yml down
docker image remove $TAG
docker build -f docker/dockerfile --no-cache -t $TAG .

read -rsp "GHCR_TOKEN: " GHCR_TOKEN
echo
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u alan-venv --password-stdin
docker push $TAG
docker logout ghcr.io

clear
echo "Version: $VERSION"
