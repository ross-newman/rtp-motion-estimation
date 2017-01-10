/*
 * inference-101
 */
#include "config.h"
#include "cudaYUV.h"


__constant__ uint32_t constAlpha;
__constant__ float  constHueColorSpaceMat[9];

#define LIMIT_RGB(x)    (((x)<0)?0:((x)>255)?255:(x))

__device__ void YUV82RGB(uint8_t *yuvi, uint8_t *red, uint8_t *green, uint8_t *blue)
{
	const float y0 = float(yuvi[0]);
	const float cb0    = float(yuvi[1]);
	const float cr0    = float(yuvi[2]);

//printf("YUV luma=%f, u=%f, v=%f\n", y0, cb0, cr0);

	unsigned char r0(LIMIT_RGB(round((298.082 * y0) / 256.0 + (408.583 * cr0) / 256.0 - 222.921)));


	unsigned char g0(LIMIT_RGB(round((298.082 * y0) / 256.0 - (100.291 * cb0) / 256.0
			- (208.120 * cr0) / 256.0 + 135.576)));


	unsigned char b0(LIMIT_RGB(round((298.082 * y0) / 256.0 + (516.412 * cb0) / 256.0 - 276.836)));

	*red    = (uint8_t)r0;
	*green  = (uint8_t)g0;
	*blue   = (uint8_t)b0;
}

//-------------------------------------------------------------------------------------------------------------------------
// RTP YUV color space conversion

__global__ void YUVToRGBA(uint8_t* srcImage,  size_t nSourcePitch,
                           uint8_t* dstImage,     size_t nDestPitch,
                           uint32_t width,       uint32_t height)
{
    int x, y;
    uint32_t processingPitch = ((width) + 63) & ~63;
    uint8_t *srcImageU8 = srcImage;

    processingPitch = nSourcePitch * 2;

    // Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
    x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
    y = blockIdx.y *  blockDim.y       +  threadIdx.y;

    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;
//printf("pixel x=%x, y=%d, width=%d, height=%d, RGBX@%d\n",x,y,width, height, y * (width *4) + x*4);
    // this steps performs the color conversion
    uint8_t yuvi[6];
    uint8_t red[2], green[2], blue[2];
#if GSTREAMER
    int offset[6] = {  1,  0,  2,  3,  0,  2};
#else
    //                Y0  Cb  Cr  Y1  Cb  Cr
    int offset[6] = {  1,  0,  2,  3,  0,  2};
#endif

	for (int c=0;c<6;c++)
	{
		yuvi[c] = srcImageU8[y * processingPitch + x*2 + offset[c]];
	}

    // YUV to RGB Transformation conversion
    YUV82RGB(&yuvi[0], &red[0], &green[0], &blue[0]);
    YUV82RGB(&yuvi[3], &red[1], &green[1], &blue[1]);
#if 1
	dstImage[y * (width *4) + x*4]      = red[0];
	dstImage[y * (width *4) + x*4 + 1]  = green[0];
	dstImage[y * (width *4) + x*4 + 2]  = blue[0];
	dstImage[y * (width *4) + x*4 + 3]  = 0xff;
	dstImage[y * (width *4) + x*4 + 4]  = red[1];
	dstImage[y * (width *4) + x*4 + 5]  = green[1];
	dstImage[y * (width *4) + x*4 + 6]  = blue[1];
	dstImage[y * (width *4) + x*4 + 7]  = 0xff;
#else
	dstImage[y * (width *3) + x*3]      = red[0];
	dstImage[y * (width *3) + x*3 + 1]  = green[0];
	dstImage[y * (width *3) + x*3 + 2]  = blue[0];
	dstImage[y * (width *3) + x*3 + 3]  = red[1];
	dstImage[y * (width *3) + x*3 + 4]  = green[1];
	dstImage[y * (width *3) + x*3 + 5]  = blue[1];
#endif
}

// cudaYUVToRGBA
cudaError_t cudaYUVToRGBA( uint8_t* srcDev, size_t srcPitch, uint8_t* destDev, size_t destPitch, size_t width, size_t height )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	if( srcPitch == 0 || destPitch == 0 || width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const dim3 blockDim(8,8,1);

	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height, blockDim.y), 1);

	YUVToRGBA<<<gridDim, blockDim>>>( (uint8_t*)srcDev, srcPitch, destDev, destPitch, width, height );

	return CUDA(cudaGetLastError());
}

cudaError_t cudaYUVToRGBA( uint8_t* srcDev, uint8_t* destDev, size_t width, size_t height )
{
	cudaYUVToRGBA(srcDev, width * sizeof(uint8_t), destDev, width * sizeof(uint8_t) * 4, width, height);
	return cudaSuccess;
}

//-------------------------------------------------------------------------------------------------------------------------
// RTP YUV color space conversion

__device__ void RGBToYUV(bool even, uint8_t *rgb, uint8_t *Y, uint8_t *U, uint8_t *V)
{
	int R,G,B;

	// Set inital values to zero
    *Y=0;
    *U=0;
    *V=0;

    // Get the RGB Values
	B=rgb[0];
	G=rgb[1];
	R=rgb[2];
	/* sample luma for every pixel */
	*Y  = LIMIT_RGB((0.257 * R) + (0.504 * G) + (0.098 * B) + 16);

	if (even == true)
	{
		*V =  LIMIT_RGB((0.439 * R) - (0.368 * G) - (0.071 * B) + 128);
	}
	else
	{
		*U = LIMIT_RGB(-(0.148 * R) - (0.291 * G) + (0.439 * B) + 128);
	}
}

__global__ void RGBAToYUV(uint8_t* srcImage,  size_t nSourcePitch,
                           uint8_t* dstImage,     size_t nDestPitch,
                           uint32_t width,       uint32_t height)
{
    int x, y;
    uint32_t processingPitch = ((width) + 63) & ~63;
    uint8_t *srcImageU8 = srcImage;

    processingPitch = nDestPitch;

    // Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
    x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
    y = blockIdx.y *  blockDim.y       +  threadIdx.y;

    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

//printf("YUV >> x=%d, y=%d, pitch=%d, dest=0x%x\n", x, y, processingPitch, dstImage);

    // this steps performs the color conversion
    uint8_t *rgb;
    rgb = &srcImageU8[y * (width * 4) + x*4];
    uint8_t Y[2], u[2], v[2];

    // Process u and v for odd/even pixels
    RGBToYUV(true, &rgb[0], &Y[0], &u[0], &v[0]);
    RGBToYUV(false, &rgb[4], &Y[1], &u[1], &v[1]);

    // Check these values have not been set
    if (u[0] != 0) return;
    if (v[1] != 0) return;

#if 1
    // Pack the converted data into the destination buffer
	dstImage[y * processingPitch + x*2 ]     = v[0];
	dstImage[y * processingPitch + x*2 + 1]  = Y[0];
	dstImage[y * processingPitch + x*2 + 2]  = u[1];
	dstImage[y * processingPitch + x*2 + 3]  = Y[1];
#else
	dstImage[y * processingPitch + x*2 ]     = 0xff;
	dstImage[y * processingPitch + x*2 + 1]  = 0xff;
	dstImage[y * processingPitch + x*2 + 2]  = 0xff;
	dstImage[y * processingPitch + x*2 + 3]  = 0xff;
#endif
}

// cudaRGBAToYUV
cudaError_t cudaRGBAToYUV( uint8_t* srcDev, size_t srcPitch, uint8_t* destDev, size_t destPitch, size_t width, size_t height )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	if( srcPitch == 0 || destPitch == 0 || width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height, blockDim.y), 1);
//printf("########### srcDev 0x%x, srcPitch %d, destDev 0x%x, destPitch %d, width %d, height %d\n", srcDev, srcPitch, destDev, destPitch, width, height);
	RGBAToYUV<<<gridDim, blockDim>>>( (uint8_t*)srcDev, srcPitch, destDev, destPitch, width, height );

	return CUDA(cudaGetLastError());
}


cudaError_t cudaRGBAToYUV( uint8_t* srcDev, uint8_t* destDev, size_t width, size_t height )
{
	cudaRGBAToYUV(srcDev, width * sizeof(uint8_t) * 4, destDev, width * sizeof(uint8_t) * 2, width, height);
	return cudaSuccess;
}

__global__ void RGBToYUV(uint8_t* srcImage,  size_t nSourcePitch,
                           uint8_t* dstImage,     size_t nDestPitch,
                           uint32_t width,       uint32_t height)
{
    int x, y;
    uint32_t processingPitch = ((width) + 63) & ~63;
    uint8_t *srcImageU8 = srcImage;

    processingPitch = nDestPitch;

    // Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
    x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
    y = blockIdx.y *  blockDim.y       +  threadIdx.y;

    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

//printf("YUV >> x=%d, y=%d, pitch=%d, dest=0x%x\n", x, y, processingPitch, dstImage);

    // this steps performs the color conversion
    uint8_t *rgb;
    rgb = &srcImageU8[(y * (width * 3) + x*3)];
    uint8_t Y[2], u[2], v[2];

    // Process u and v for odd/even pixels
    RGBToYUV(true, &rgb[0], &Y[0], &u[0], &v[0]);
    RGBToYUV(false, &rgb[3], &Y[1], &u[1], &v[1]);

    // Check these values have not been set
    if (u[0] != 0) return;
    if (v[1] != 0) return;

    // Pack the converted data into the destination buffer
	dstImage[y * processingPitch + x*2 ]     = v[0];
	dstImage[y * processingPitch + x*2 + 1]  = Y[0];
	dstImage[y * processingPitch + x*2 + 2]  = u[1];
	dstImage[y * processingPitch + x*2 + 3]  = Y[1];
}

// cudaRGBToYUV
cudaError_t cudaRGBToYUV( uint8_t* srcDev, size_t srcPitch, uint8_t* destDev, size_t destPitch, size_t width, size_t height )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	if( srcPitch == 0 || destPitch == 0 || width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height, blockDim.y), 1);

	RGBToYUV<<<gridDim, blockDim>>>( (uint8_t*)srcDev, srcPitch, destDev, destPitch, width, height );

	return CUDA(cudaGetLastError());
}

cudaError_t cudaRGBToYUV( uint8_t* srcDev, uint8_t* destDev, size_t width, size_t height )
{
	cudaRGBToYUV(srcDev, width * sizeof(uint8_t) * 3, destDev, width * sizeof(uint8_t) * 2, width, height);
	return cudaSuccess;
}


