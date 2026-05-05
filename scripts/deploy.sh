#!/bin/bash

docker compose -f docker/docker-compose.yml down
docker image remove ghcr.io/alan-venv/rinha-service-2026:latest
docker build -f docker/dockerfile -t ghcr.io/alan-venv/rinha-service-2026:latest .

read -rsp "GHCR_TOKEN: " GHCR_TOKEN
echo
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u alan-venv --password-stdin
docker push ghcr.io/alan-venv/rinha-service-2026:latest
docker logout ghcr.io
