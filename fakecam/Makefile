all: .venv /dev/video20
	.venv/bin/python fake.py

.venv:
	virtualenv .venv
	.venv/bin/pip install -r requirements.txt

/dev/video20:
	sudo modprobe -r v4l2loopback
	sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="v4l2loopback" exclusive_caps=1