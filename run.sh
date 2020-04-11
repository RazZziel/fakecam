#!/bin/bash

make -C bodypix/ &
make -C fakecam/

kill $(jobs -p)