#!/bin/bash
echo "" > log-compose
echo "" > log-err-compose
for f in docker-compose_*; do
	echo "docker-compose for ${f}"
	docker-compose -f $f up --build >log-compose 2>log-err-compose
done
