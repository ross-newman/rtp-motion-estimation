#ifndef __CONFIG_H
#define __CONFIG_H

#define HEIGHT 480  		// Lower resolution stream for RTP
#define WIDTH 640

#define HEADLESS 0  		// Dont put anything out on the local display
#define GST_MULTICAST 1	// Set this for multicast sources
#define RTP_STREAM_SOURCE 1 // define to use RTP Stream class as source (not gstreamer)
#define GST_SOURCE 1  		// 1 if RTP source, 0 if file source.
#define GST_RTP_SINK 1		// RTP Output
#define TIMEING_DEBUG 0		// Show timings

#endif
