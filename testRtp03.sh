#!/bin/bash
export HEIGHT=720
export WIDTH=1280
#export HEIGHT=480
#export WIDTH=640
pkill gst-launch-1.0
echo ### Launching LIVE stream...
gst-launch-1.0 -v videotestsrc ! video/x-raw, format=RGB, framerate=25/1, width=${WIDTH}, height=${HEIGHT} ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=5004 
echo ### LIVE stream ended...

