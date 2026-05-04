#!/bin/bash

rm resources/*.bin
cargo run --bin map
cargo run --bin ivf
