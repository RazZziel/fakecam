all: .venv /dev/video20
	.venv/bin/python fake.py $$ARGS

.venv:
	virtualenv .venv
	.venv/bin/pip install -r requirements.txt

/dev/video20:
	sudo modprobe -r v4l2loopback
	sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="v4l2loopback" exclusive_caps=1


bodypix.json:
	wget https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/quant2/075/model-stride16.json -O bodypix.json
	.venv/bin/pip install tensorflowjs
	# TODO: https://github.com/tensorflow/tfjs/tree/master/tfjs-converter
	# TODO: https://github.com/ajaichemmanam/simple_bodypix_python/blob/master/evalbody_singleposemodel.py