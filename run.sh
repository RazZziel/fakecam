#!/bin/bash

make -C bodypix/ &
make -C fakecam/ ARGS="$*"

kill $(jobs -p)