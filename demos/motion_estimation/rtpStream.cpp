#include <pthread.h>
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

rtpStream::rtpStream(int height, int width) :
    camera(height, width)
{
    mFrame = 0;
    pthread_mutex_init(&mutex, NULL);
}

/* Broadcast the stream to port 5004 */
void rtpStream::rtpStreamIn( char* hostname, int portno)
{
	mPortNoIn = portno;
	strcpy(mHostnameIn, hostname);
}

void rtpStream::rtpStreamOut(char* hostname, int portno)
{
	// TODO : Implement input
	mPortNoOut = portno;
	strcpy(mHostnameOut, hostname);
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
    mServer = gethostbyname(mHostnameOut);
    if (mServer == NULL) {
        fprintf(stderr,"ERROR, no such host as %s\n", mHostnameOut);
        exit(0);
    }

    /* build the server's Internet address */
    bzero((char *) &mServeraddr, sizeof(mServeraddr));
    mServeraddr.sin_family = AF_INET;
    bcopy((char *)mServer->h_addr,
	  (char *)&mServeraddr.sin_addr.s_addr, mServer->h_length);
    mServeraddr.sin_port = htons(mPortNoOut);

    /* send the message to the server */
    mServerlen = sizeof(mServeraddr);

    return true;
}

void rtpStream::Close()
{
	close(mSockfd);
}

#if ARM
void endianswap32(uint32_t *data, int length)
{
  int c = 0;
  for (c=0;c<length;c++)
    data[c] = __bswap_32 (data[c]);
}

void endianswap16(uint16_t *data, int length)
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

void *TransmitThread(void* data)
{
    rtp_packet packet;
	tx_data *arg;
    char *yuv;
    int c=0;
    int n=0;

	arg = (tx_data *)data;

#if RTP_TO_YUV_ONGPU
    // Convert the whole frame into YUV
    char yuvdata[arg->width * arg->height * 2];
	ConvertRGBtoYUV((void*)arg->rgbframe, arg->gpuAddr, (void**)&yuvdata, arg->width, arg->height);
#endif

    sequence_number=0;

    /* send a frame */
    pthread_mutex_lock(&arg->stream->mutex);
    {
      struct timeval NTP_value;
      int32_t time = 10000;

      for (c=0;c<(arg->height);c++)
      {
        int x,last = 0;
        if (c==arg->height-1) last=1;
        arg->stream->update_header((header*)&packet, c, last, time, RTP_SOURCE);

#if RTP_TO_YUV_ONGPU
        x = c * (arg->width * 2);
        // Copy previously converted line into header
		memcpy(packet.data, (void*)&yuvdata[x], arg->width * 2);
#else
        x = c * (arg->width * PITCH);
        // CPU conversion, might as well do one line at a time
        rgbtoyuv(mHeight, mWidth, packet.data, (char*)&args.rgbframe[x]);
#endif

#if ARM
        endianswap32((uint32_t *)&packet, sizeof(rtp_header)/4);
        endianswap16((uint16_t *)&packet.head.payload, sizeof(payload_header)/2);
#endif
        n = sendto(arg->stream->mSockfd, (char *)&packet, sizeof(rtp_packet), 0, (const sockaddr*)&arg->stream->mServeraddr, arg->stream->mServerlen);
        if (n < 0)
          fprintf(stderr, "ERROR in sendto");
      }

      // printf("Sent frame %d\n", mFrame++);
    }
    pthread_mutex_unlock(&arg->stream->mutex);
}

// Arguments sent to thread
static tx_data args;
int rtpStream::Transmit(char* rgbframe, bool gpuAddr)
{
	pthread_t tx;
	args.rgbframe = rgbframe;
	args.gpuAddr = gpuAddr;
	args.width = mWidth;
	args.height = mHeight;
	args.stream = this;

	// Start a thread so we can start capturing the next frame while transmitting the data
	pthread_create(&tx, NULL, TransmitThread, &args );

    return 0;
}

