#!/bin/bash

mkdir -p graphs
rm graphs/*
cargo run -- -g graphs "$@" 2> /dev/null 
dot -Tsvg -O graphs/*.dot && firefox graphs/*.svg
