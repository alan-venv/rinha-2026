#!/bin/bash

NAME="ghcr.io/alan-venv/rinha-service-2026"
VERSION=`date +%s`
TAG="$NAME:${VERSION}"

docker compose -f docker/docker-compose.yml down
docker rmi $(docker images --filter=reference="$NAME" -q)
docker build -f docker/dockerfile --no-cache -t $TAG .

clear
read -rsp "GHCR_TOKEN: " GHCR_TOKEN
echo
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u alan-venv --password-stdin
docker push $TAG
docker logout ghcr.io

clear
echo "Version: $VERSION"
