#!/bin/bash

docker compose -f docker/docker-compose.yml down > /dev/null 2>&1
docker image remove ghcr.io/alan-venv/rinha-service-2026:test > /dev/null 2>&1
docker build -f docker/dockerfile -t ghcr.io/alan-venv/rinha-service-2026:test . > /dev/null 2>&1
docker compose -f docker/docker-compose.yml up -d > /dev/null 2>&1
k6 run scripts/k6/main.js > /dev/null 2>&1
docker compose -f docker/docker-compose.yml down > /dev/null 2>&1

cat scripts/k6/results.json | grep -e "ms" -e "false_" -e "final" | tr -d '[:blank:]'
