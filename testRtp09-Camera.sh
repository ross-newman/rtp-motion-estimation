#!/bin/bash
export HEIGHT=480
export WIDTH=640
#export HEIGHT=480
#export WIDTH=640
pkill gst-launch-1.0
echo ### Launchin player
gst-launch-1.0 -v udpsrc address=239.192.1.44 port=5005 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)RAW, sampling=(string)YCbCr-4:2:2, depth=(string)8, width=(string)${WIDTH}, height=(string)${HEIGHT}, payload=(int)96" ! queue ! rtpvrawdepay ! queue ! xvimagesink sync=false 

