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
#define HEADLESS 1  		// Dont put anything out on the local display
#define RTP_MULTICAST 0 	// Set this for multicast sources
#define RTP_STREAM_SOURCE 1 // define to use RTP Stream class as source (not gstreamer)
#define GST_SOURCE 1  		// 1 if RTP source, 0 if file source.
#define GST_RTP_SINK 1		// RTP Output
#define TIMEING_DEBUG 0		// Show timings

// 
// Connection details
//
//#define IP_MULTICAST_IN     "239.192.2.40"
//#define IP_MULTICAST_OUT    "239.192.1.198"
#define IP_PORT_IN			5004
#define IP_PORT_OUT   		5005
#define IP_RTP_IN       "127.0.0.1"
#define IP_RTP_OUT      "127.0.0.1"

//
// Motion params
//
#define MOTION_SPEED_THRESHOLD 0


//
// Gstreamer1.0 compatability
//
#define GST_1_FUDGE 1       // Offset date by 1 byte as gstream has an RTP bug. Must have RTP_STREAM_SOURCE=1

#endif
