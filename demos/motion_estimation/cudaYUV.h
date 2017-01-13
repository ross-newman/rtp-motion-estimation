/*
 * http://github.com/dusty-nv/jetson-inference
 */

#ifndef __CUDA_YUV_CONVERT_H
#define __CUDA_YUV_CONVERT_H

#include <stdint.h>
#include <NVX/nvx.h>
#include <VX/vx_types.h>
#include "cudaUtility.h"

/**
 * Setup NV12 color conversion constants.
 * cudaNV12SetupColorspace() isn't necessary for the user to call, it will be
 * called automatically by cudaNV12ToRGBA() with a hue of 0.0.
 * However if you want to setup custom constants (ie with a hue different than 0),
 * then you can call cudaNV12SetupColorspace() at any time, overriding the default.
 */
cudaError_t cudaNV12SetupColorspace( float hue = 0.0f );

//////////////////////////////////////////////////////////////////////////////////
/// @name YUV to RGBf
//////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaYUVToRGBA( uint8_t* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height );
cudaError_t cudaYUVToRGBA( uint8_t* input, uint8_t* output, size_t width, size_t height );

cudaError_t cudaRGBAToYUV( uint8_t* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height );
cudaError_t cudaRGBAToYUV( uint8_t* input, uint8_t* output, size_t width, size_t height );

cudaError_t cudaRGBToYUV( uint8_t* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height );
cudaError_t cudaRGBToYUV( uint8_t* input, uint8_t* output, size_t width, size_t height );

cudaError_t cudaMotionFields( uint8_t* image, vx_float32* motionfeilds, size_t width, size_t height);

///@}

#endif

