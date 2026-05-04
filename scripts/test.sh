#!/bin/bash

docker compose -f docker/docker-compose.yml down
docker image remove ghcr.io/alan-venv/rinha-service-2026:latest
docker build -f docker/dockerfile -t ghcr.io/alan-venv/rinha-service-2026:latest .
docker compose -f docker/docker-compose.yml up -d
k6 run scripts/k6/main.js
docker compose -f docker/docker-compose.yml down

clear
cat scripts/k6/results.json | grep -e "ms" -e "value" -e "false_" -e "final" | tr -d '[:blank:]'
