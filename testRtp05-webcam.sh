#!/bin/bash
export HEIGHT=480
export WIDTH=640
#export HEIGHT=480
#export WIDTH=640
pkill gst-launch-1.0

echo ### Launching LIVE stream...
gst-launch-1.0 -v videotestsrc horizontal-speed=1 ! video/x-raw, format=UYVY, framerate=25/1, width=${WIDTH}, height=${HEIGHT} ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=5004 &

echo ### Launchin player
gst-launch-1.0 -v udpsrc port=5005 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)RAW, sampling=(string)YCbCr-4:2:2, depth=(string)8, width=(string)${WIDTH}, height=(string)${HEIGHT}, payload=(int)96" ! queue ! rtpvrawdepay ! queue ! xvimagesink sync=false 

