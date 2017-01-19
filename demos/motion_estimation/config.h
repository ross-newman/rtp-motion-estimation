#ifndef __CONFIG_H
#define __CONFIG_H

//
// Default resolution for streams
//
#define HEIGHT 480  		// Lower resolution stream for RTP
#define WIDTH 640

//
// Application configuration
//
#define HEADLESS 0  		// Dont put anything out on the local display
#define RTP_MULTICAST 0		// Set this for multicast sources
#define RTP_STREAM_SOURCE 1 // define to use RTP Stream class as source (not gstreamer)
#define GST_SOURCE 1  		// 1 if RTP source, 0 if file source.
#define GST_RTP_SINK 0		// RTP Output
#define TIMEING_DEBUG 0		// Show timings

// 
// Connection details
//
#define IP_MULTICAST_IN     "239.192.1.43" 
#define IP_MULTICAST_OUT    "239.192.1.198"
#define IP_UNICAST          "127.0.0.1"

//
// Gstreamer1.0 compatability
//
#define GST_1_FUDGE 1       // Offset date by 1 byte as gstream has an RTP bug.

#endif
