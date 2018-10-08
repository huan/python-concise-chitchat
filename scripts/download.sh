#!/usr/bin/env bash
set -e

[ -d data ] || mkdir data

(
    cd data

    [ -e cornell_movie_dialogs_corpus.zip ] || curl -L -O -C - http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
    [ -d 'cornell movie-dialogs corpus' ]   || unzip -o cornell_movie_dialogs_corpus.zip

    [ -e glove.6B.50d.txt.gz ]  || curl -L -O -C - https://github.com/zixia/concise-chit-chat/releases/download/v0.0.1/glove.6B.50d.txt.gz
    [ -e glove.6B.50d.txt ]     || gzip -d -f -k glove.6B.50d.txt.gz
)
