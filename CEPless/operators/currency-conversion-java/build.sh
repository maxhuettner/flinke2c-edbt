#!/bin/bash

mvn compile assembly:single

# amd64 for the NodeManager hosts, arm64 for local builds
docker buildx build --platform linux/amd64,linux/arm64 -t $1/$2 --push .
