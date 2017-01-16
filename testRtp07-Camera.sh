#!/bin/bash
export HEIGHT=480
export WIDTH=640
#export HEIGHT=480
#export WIDTH=640
pkill gst-launch-1.0

gst-launch-1.0 -v v4l2src do-timestamp=true io-mode=1 device=/dev/video0 ! \
	videoscale ! \
	videoconvert  ! \
	"video/x-raw, width=${WIDTH}, height=${HEIGHT}, format=(string)UYVY" ! \
	queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=5004 &

echo ### Launchin player
gst-launch-1.0 -v udpsrc  port=5005 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)RAW, sampling=(string)YCbCr-4:2:2, depth=(string)8, width=(string)${WIDTH}, height=(string)${HEIGHT}, payload=(int)96" ! queue ! rtpvrawdepay ! queue ! xvimagesink  

