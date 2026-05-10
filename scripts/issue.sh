#!/usr/bin/env bash

set -euo pipefail

owner="zanfranceschi"
repo="rinha-de-backend-2026"
title="rinha/test"
body="rinha/test"

read -r -s -p "GitHub token: " github_token
echo

if [ -z "$github_token" ]; then
  echo "erro"
  exit 1
fi

response_file="$(mktemp)"
trap 'rm -f "$response_file"' EXIT

http_code="$(curl -s -o "$response_file" -w "%{http_code}" -L \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $github_token" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/$owner/$repo/issues" \
  -d "$(printf '{"title":"%s","body":"%s"}' "$title" "$body")")"

if [ "$http_code" = "201" ]; then
  issue_url="$(sed -n 's/.*"html_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$response_file" | head -n 1)"
  if [ -n "$issue_url" ]; then
    echo "$issue_url"
  else
    echo "erro"
    exit 1
  fi
else
  echo "erro"
  exit 1
fi
