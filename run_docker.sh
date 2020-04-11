#!/bin/bash

# load fake webcam kernel module
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="v4l2loopback" exclusive_caps=1

# create a network
docker network rm fakecam
docker network create --driver bridge fakecam

docker build -t bodypix ./bodypix

# start the bodypix app
docker rm bodypix
docker run -d \
  --name=bodypix \
  --network=fakecam \
  --restart always \
  -p 9000:9000 \
  `#--gpus=all` --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  bodypix

docker build -t fakecam ./fakecam

# start the camera, note that we need to pass through video devices,
# and we want our user ID and group to have permission to them
# you may need to `sudo groupadd $USER video`
docker rm fakecam
docker run \
  --name=fakecam \
  --network=fakecam \
  -u "$(id -u):$(getent group video | cut -d: -f3)" \
  $(find /dev -name 'video*' -printf "--device %p ") \
  fakecam

docker kill bodypix