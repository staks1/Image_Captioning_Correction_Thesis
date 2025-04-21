#!/bin/bash

mkdir -p Models
chmod -R  -R $(id -u):$(id -g) Models
chmod 777 Models
docker-compose up