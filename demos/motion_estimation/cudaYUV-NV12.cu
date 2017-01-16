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

    // Pitch is four as data is RGBX
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

#define ARROW_PITCH 16
__device__ unsigned int motionDetect(uint8_t* image, uint32_t* motionfeilds, uint8_t x, uint8_t y)
{
	int processingPitch = 320 * 2;
	int motionStride = ARROW_PITCH/2;
	uint32_t sum = 0;
	int xx,yy = 0;

//printf("motionDetect >> x=%d, y=%d\n", x, y);

	// Calculate starting position in image
	y = y*processingPitch;
	x = x*motionStride;

	// Sum values in area (8x8 as motion (two vx_float32 values) is for each 2x2 pixels)
	for (yy=y; yy<y+(processingPitch*motionStride); yy+=processingPitch)
	{
		for (xx=x; xx<x+motionStride; xx++)
		{
printf("motionDetect >> xx=%d, yy=%d\n", xx, yy);
//			sum += motionfeilds[xx + yy];
		}
	}
	return sum;
}

/*
 1 2 3 4 5 6 7 8 9 0 a b c d e f
1
2
3
4
5
6              *
7            *
8          *
9        * * * * * * * * * *
0          *
a            *
b              *
c
d
e
f
*/
__device__ char arrow_left[16][16] = {
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0},
{0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
};


/*
 1 2 3 4 5 6 7 8 9 0 a b c d e f
1
2
3
4
5
6          * * * * *
7          * *
8          *   *
9          *     *
0          *       *
a                    *
b                      *
c
d
e
f
*/
__device__ char arrow_inbetween[16][16] = {
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0},
{0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0},
{0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0},
{0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
};

enum edir
{
	DIR_N,
	DIR_NE,
	DIR_E,
	DIR_SE,
	DIR_S,
	DIR_SW,
	DIR_W,
	DIR_NW,
} edir;

__device__ bool renderArrow(uint8_t* image, int direction, int speed, uint32_t x, uint32_t y)
{
	int processingPitch = 640*4;
	uint32_t xx,yy = 0;
	int pixx = 0;
	int pixy = 0;
	uint32_t base;
	int arrowType = 0;

	// Decide what arrow we need to draw
	switch (direction)
	{
	case DIR_N :
	case DIR_W :
	case DIR_S :
	case DIR_E :
		arrowType = 1;
		break;
	case DIR_NW :
	case DIR_SW :
	case DIR_NE :
	case DIR_SE :
		arrowType = 2;
		break;
	default : // Unable to draw at the moment
		break;
	}

	// Calculate starting position in image
	y = y*processingPitch*ARROW_PITCH;
	x = x*ARROW_PITCH*4;
	// Sum values in area (8x8 as motion is for each 2x2 pixels)
	for (yy=y; yy<y+(processingPitch*ARROW_PITCH); yy+=processingPitch)
	{
		pixx=0;
		for (xx=x; xx<x+(ARROW_PITCH*4); xx+=4)
		{
			int mapx,mapy;
			switch (direction)
			{
			case DIR_N :
				mapx = pixx++;
				mapy = pixy;
				break;
			case DIR_W :
				mapx = pixy;
				mapy = pixx++;
				break;
			case DIR_S :
				mapx = pixx++;
				mapy = 16-pixy;
				break;
			case DIR_E :
				mapx = pixy;
				mapy = 16-pixx++;
				break;
			case DIR_NW :
				mapx = pixx++;
				mapy = pixy;
				break;
			case DIR_SE :
				mapx = 16-pixy;
				mapy = 16-pixx++;
				break;
			case DIR_SW :
				mapx = 16-pixy;
				mapy = pixx++;
				break;
			case DIR_NE :
				mapx = pixy;
				mapy = 16-pixx++;
				break;
			default : // Unable to draw at the moment
				break;
			}

			switch (arrowType)
			{
			case 1 :
				// Draw the arrow N, S, E, W
				if (arrow_left[mapx][mapy] == 1) // draw pixel from map
				{
					base = yy + xx;
					image[base] = 0x00;
					image[base+1] = 0xff;
					image[base+2] = 0xff;
				}
				break;
			case 2 :
				// Draw the arrow NE, NW, SE, SW
				if (arrow_inbetween[mapx][mapy] == 1) // draw pixel from map
				{
					base = yy + xx;
					image[base] = 0x00;
					image[base+1] = 0xff;
					image[base+2] = 0xff;
				}
				break;
			}
		}
		pixy++;
	}
	return true;
}


#include <math.h>


typedef struct Vector2
{
    float X;
    float Y;
};

// Normalizes the 2D vector
__device__ Vector2 Normalize(float X, float Y){
    Vector2 vector;
    float length;
    length = sqrt(X * X + Y * Y);

    if(length != 0){
        vector.X = X/length;
        vector.Y = Y/length;
    }
    return vector;
}

// Converts radians to degrees
__device__ float Rad2Deg(float radians){
    return radians*(180/3.141592653589793238);
}

__device__ float calculateAngle(vx_float32 dir_x, vx_float32 dir_y)
{
    Vector2 vector2D = Normalize(dir_x, dir_y);

    // Calculate angle for the 2D vector
    float angle = atan2(vector2D.X,vector2D.Y);
    angle = Rad2Deg(angle);

	if (angle < 0)
		angle = 180 + (180 + angle);
    return angle;
}

__device__ bool render(uint8_t* image, vx_float32 *motion, uint32_t x, uint32_t y)
{
	vx_float32 dir_x, dir_y;
	int direction, speed = 5;
	int n = 45 / 2;
	float angle;

	dir_x = motion[0];
	dir_y = motion[1];

	if ((dir_x != 0) || (dir_y != 0))
	{
		angle = calculateAngle(dir_x,dir_y);

//		printf("(2D) Angle from X-axis: %f\n",angle);
		direction = DIR_N;
		if ((angle > 0-n) && (angle < n))
			direction = DIR_S;
		else if ((angle >= 180-n) && (angle < 180+n))
			direction = DIR_N;
		else if ((angle >= 90-n) && (angle < 90+n))
			direction = DIR_E;
		else if ((angle >= 270-n) && (angle < 270+n))
			direction = DIR_W;
		else if ((angle >= n) && (angle < 90-n))
			direction = DIR_SE;
		else if ((angle >= 90+n) && (angle < 180-n))
			direction = DIR_NE;
		else if ((angle >= 180+n) && (angle < 270-n))
			direction = DIR_NW;
		else if ((angle >= 270+n) && (angle < 360-n))
			direction = DIR_SW;
		else
			direction = 999;
		renderArrow(image, direction, speed, x, y);
	}

	return true;
}

__global__ void MotionFields(uint8_t* image,
                           vx_float32* motionfeilds,
                           uint32_t width,
                           uint32_t height)
{
    int x, y;
	vx_float32 *motion = 0;
	uint32_t processingPitch = width/2 * 2;

    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

//printf("Motion >> x=%d, y=%d\n", x, y);

    // Grab motion for our pixel location (location of arrow)
    motion = &motionfeilds[(y*processingPitch)*(ARROW_PITCH/2) + x * (ARROW_PITCH)];

	render(image, motion, x, y);
}

static vx_float32* motionfeildsGPU = 0;

cudaError_t cudaMotionFields( uint8_t* image, vx_float32* motionfeilds, size_t width, size_t height)
{
	int mfsize =  (width/2 * height/2) * sizeof(vx_float32) * 2;
	if( !image || !motionfeilds )
		return cudaErrorInvalidDevicePointer;

	// Push the motion feild data to the GPU
	if( !motionfeildsGPU )
	{
		if( CUDA_FAILED(cudaMalloc(&motionfeildsGPU, mfsize) ) )
		{
			printf(LOG_CUDA "cudaMotionFields -- failed to allocate memory for %ux%u RGBA texture\n", width/2, height/2);
			return CUDA(cudaGetLastError());
		}
	}
	// HOST buffer so copy the RGB data to the GPU
	cudaMemcpy( motionfeildsGPU, motionfeilds, mfsize, cudaMemcpyHostToDevice );

	const dim3 blockDim(8,8,1);
	// 32 x 32 blocks for arrows
	const dim3 gridDim(iDivUp(width/ARROW_PITCH,blockDim.x), iDivUp(height/ARROW_PITCH, blockDim.y), 1);

	MotionFields<<<gridDim, blockDim>>>( image, motionfeildsGPU, width, height );

	return CUDA(cudaGetLastError());
}


