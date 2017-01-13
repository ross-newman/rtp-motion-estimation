#include "config.h"
#include "rtpStream.h"

extern void DumpHex(const void* data, size_t size);

#define RTP_TO_YUV_ONGPU 1 // Offload colour conversion to GPU if set
#define PITCH 4
#if RTP_TO_YUV_ONGPU
#include "cudaYUV.h"

void * mRGBA;
void * mYUV;
// ConvertRGBA

bool ConvertRGBtoYUV( void* input, bool gpuAddr, void** output, size_t width, size_t height )
{
	if( !input || !output )
		return false;

	if (gpuAddr)
	{
		// Data is already on the GPU so no need to copy
		mRGBA = input;
	}
	else
	{
		if( !mRGBA )
		{
			if( CUDA_FAILED(cudaMalloc(&mRGBA, (width * height) * 4)) )
			{
				printf(LOG_CUDA "rtpStream -- failed to allocate memory for %ux%u RGBA texture\n", (unsigned int)width, (unsigned int)height);
				return false;
			}
		}
		// HOST buffer so copy the RGB data to the GPU
		cudaMemcpy( mRGBA, input, (width * height) * PITCH, cudaMemcpyHostToDevice );
	}

	if( !mYUV )
	{
		if( CUDA_FAILED(cudaMalloc(&mYUV, (width * height) * 2)) )
		{
			printf(LOG_CUDA "rtpStream -- failed to allocate memory for %ux%u YUV texture\n", (unsigned int)width, (unsigned int)height);
			return false;
		}
	}

	// RTP is YUV

	// Push the RGB data over to the GPU
	if( CUDA_FAILED(cudaRGBAToYUV((uint8_t*)mRGBA, (uint8_t*)mYUV, (size_t)width, (size_t)height)) )

	{
		return false;
	}

    // Pull the color converted YUV data off the GPU
	cudaMemcpy( output, mYUV, (width * height) * 2, cudaMemcpyDeviceToHost );

	return true;
}

#else
typedef struct float4 {
    float x;
    float y;
    float z;
    float w;
} float4;
#endif

/*
 * error - wrapper for perror
 */
void error(char *msg) {
    perror(msg);
    exit(0);
}

void rgbtoyuv(int y, int x, char* yuv, char* rgb)
{
  int c,cc,R,G,B,Y,U,V;
  int size;

  cc=0;
  size = x*PITCH;
  for (c=0;c<size;c+=PITCH)
  {
    R=rgb[c];
    G=rgb[c+1];
    B=rgb[c+2];
    /* sample luma for every pixel */
    Y  =      (0.257 * R) + (0.504 * G) + (0.098 * B) + 16;

    yuv[cc+1]=Y;
    if (c % 2 != 0)

    {
        V =  (0.439 * R) - (0.368 * G) - (0.071 * B) + 128;
        yuv[cc]=V;
    }
    else
    {
        U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128;
        yuv[cc]=U;
    }
    cc+=2;
  }
}

/* Broadcast the stream to port 5004 */
rtpStream::rtpStream(int height, int width, char* hostname, int portno) :
    camera(height, width)
{
	strcpy(mHostname, hostname);
    mPortNo = portno;
    mFrame = 0;
}

bool rtpStream::Open()
{
    /* socket: create the socket */
    mSockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (mSockfd < 0)
    {
        printf("ERROR opening socket");
	    return error;
	}

    /* gethostbyname: get the server's DNS entry */
    mServer = gethostbyname(mHostname);
    if (mServer == NULL) {
        fprintf(stderr,"ERROR, no such host as %s\n", mHostname);
        exit(0);
    }

    /* build the server's Internet address */
    bzero((char *) &mServeraddr, sizeof(mServeraddr));
    mServeraddr.sin_family = AF_INET;
    bcopy((char *)mServer->h_addr,
	  (char *)&mServeraddr.sin_addr.s_addr, mServer->h_length);
    mServeraddr.sin_port = htons(mPortNo);

    /* send the message to the server */
    mServerlen = sizeof(mServeraddr);

    return true;
}

void rtpStream::Close()
{
	close(mSockfd);
}

#if ARM
void rtpStream::endianswap32(uint32_t *data, int length)
{
  int c = 0;
  for (c=0;c<length;c++)
    data[c] = __bswap_32 (data[c]);
}

void rtpStream::endianswap16(uint16_t *data, int length)
{
  int c = 0;
  for (c=0;c<length;c++)
    data[c] = __bswap_16 (data[c]);
}
#endif

void rtpStream::update_header(header *packet, int line, int last, int32_t timestamp, int32_t source)
{
  bzero((char *)packet, sizeof(header));
  packet->rtp.protocol = RTP_VERSION << 30;
  packet->rtp.protocol = packet->rtp.protocol | RTP_PAYLOAD_TYPE << 16;
  packet->rtp.protocol = packet->rtp.protocol | sequence_number++;
  /* leaving other fields as zero TODO Fix*/
  packet->rtp.timestamp = timestamp += (Hz90 / RTP_FRAMERATE);
  packet->rtp.source = source;
  packet->payload.extended_sequence_number = 0; /* TODO : Fix extended seq numbers */
  packet->payload.line[0].length = MAX_BUFSIZE;
  packet->payload.line[0].line_number = line;
  packet->payload.line[0].offset = 0;
  if (last==1)
  {
    packet->rtp.protocol = packet->rtp.protocol | 1 << 23;
  }
#if 0
  printf("0x%x, 0x%x, 0x%x \n", packet->rtp.protocol, packet->rtp.timestamp, packet->rtp.source);
  printf("0x%x, 0x%x, 0x%x \n", packet->payload.line[0].length, packet->payload.line[0].line_number, packet->payload.line[0].offset);
#endif
}

int rtpStream::Transmit(char* rgbframe, bool gpuAddr)
{
    rtp_packet packet;
    char *yuv;
    int c=0;
    int n=0;

#if RTP_TO_YUV_ONGPU
    // Convert the whole frame into YUV
    char yuvdata[mWidth * mHeight * 2];
	ConvertRGBtoYUV((void*)rgbframe, gpuAddr, (void**)&yuvdata, mWidth, mHeight);
#endif

    sequence_number=0;

    /* send a frame */
    {
      struct timeval NTP_value;
      int32_t time = 10000;

      for (c=0;c<(mHeight);c++)
      {
        int x,last = 0;
        if (c==mHeight-1) last=1;
        update_header((header*)&packet, c, last, time, RTP_SOURCE);

#if RTP_TO_YUV_ONGPU
        x = c * (mWidth * 2);
        // Copy previously converted line into header
		memcpy(packet.data, (void*)&yuvdata[x], mWidth * 2);
#else
        x = c * (mWidth * PITCH);
        // CPU conversion, might as well do one line at a time
        rgbtoyuv(mHeight, mWidth, packet.data, (char*)&rgbframe[x]);
#endif

#if ARM
        endianswap32((uint32_t *)&packet, sizeof(rtp_header)/4);
        endianswap16((uint16_t *)&packet.head.payload, sizeof(payload_header)/2);
#endif
        n = sendto(mSockfd, (char *)&packet, sizeof(rtp_packet), 0, (const sockaddr*)&mServeraddr, mServerlen);
        if (n < 0)
          fprintf(stderr, "ERROR in sendto");
      }

     // printf("Sent frame %d\n", mFrame++);
    }

    return 0;
}

