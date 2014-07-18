/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#define CANNY_SHIFT 15
#define TG22        (int)(0.4142135623730950488016887242097f * (1 << CANNY_SHIFT) + 0.5f)
#define canny_push(a, b)                \
    if (mag0 > high_thr)                \
    {                                   \
        value = 2;                      \
        int c = atomic_inc(counter);    \
        stack[c] = (ushort2)(a, b);     \
    }                                   \
    else                                \
        value = 0;
    
#ifdef WITH_SOBEL

#if cn == 1
#define loadpix(addr) *(__global TYPE *)(addr)
#else
#define loadpix(addr) convert_intN(vload3(0, (__global TYPE *)(addr)))
#endif
#define storepix(value, addr) *(__global int *)(addr) = (int)(value)

/*
    stage1_with_sobel:
        Sobel operator
        Calc magnitudes
        Non maxima suppression
        Double thresholding
*/

__kernel void stage1_with_sobel(__global const uchar *src, int src_step, int src_offset, int rows, int cols,
                                __global uchar *map, int map_step, int map_offset,
                                __global ushort2 *stack, __global int *counter, 
                                int low_thr, int high_thr)
{
    int gidx_im = get_global_id(0);
    int gidy_im = get_global_id(1);

    int gidx = gidx_im - (get_group_id(0) * 2 - 1);
    int gidy = gidy_im - (get_group_id(1) * 2 - 1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

#define ptr(x, y) \
    (src + mad24(src_step, clamp(gidy + y, 0, rows - 1), mad24(clamp(gidx + x, 0, cols - 1) * cn, sizeof(TYPE), src_offset)))
    //// Sobel
    // 

    intN dx = loadpix(ptr(1, -1)) - loadpix(ptr(-1, -1)) 
                + 2 * (loadpix(ptr(1, 0)) - loadpix(ptr(-1, 0)))
                + loadpix(ptr(1, 1)) - loadpix(ptr(-1, 1));

    intN dy = loadpix(ptr(-1, 1)) - loadpix(ptr(-1, -1))
                + 2 * (loadpix(ptr(0, 1)) - loadpix(ptr(0, -1)))
                + loadpix(ptr(1, 1)) - loadpix(ptr(1, -1));

    __local int mag[18][18];

    //// Magnitude
    //

    int x, y;
#ifdef L2_GRAD
    intN magN = dx * dx + dy * dy;
#else
    intN magN = dx + dy;
#endif
#if cn == 1
    mag[lidy][lidx] = magN;
    x = dx;
    y = dy;
#else
    mag[lidy][lidx] = max(magN.x, max(magN.y, magN.z));
    if (mag[lidy][lidx] == magN.y)
    {
        dx.x = dx.y;
        dy.x = dy.y;
    }
    else if (mag[lidy][lidx] == magN.z)
    {
        dx.x = dx.z;
        dy.x = dy.z;
    }
    x = dx.x;
    y = dy.x;
#endif
    
    barrier(CLK_LOCAL_MEM_FENCE);

    //// Threshold + Non maxima suppression
    //

    int grp_idx = gidx_im / 18;
    int grp_idy = gidy_im / 18;

    gidx = clamp(gidx, grp_idx * 16, (grp_idx + 1) * 16 - 1); // Можно заменить на сдвиги
    gidy = clamp(gidy, grp_idy * 16, (grp_idy + 1) * 16 - 1);

    lidx = clamp(lidx, 1, 16);
    lidy = clamp(lidy, 1, 16);
    int mag0 = mag[lidy][lidx];

    /*
        0 - might belong to an edge
        1 - pixel doesn't belong to an edge
        2 - belong to an edge
    */
    uchar value = 1;
    if (mag0 > low_thr)
    {
        int tg22x = x * TG22;
        y <<= CANNY_SHIFT;
        int tg67x = tg22x + (x << (1 + CANNY_SHIFT));

        if (y < tg22x)
        {
            if (mag0 > mag[lidy][lidx - 1] && mag0 > mag[lidy][lidx + 1])
                canny_push(gidx, gidy)
        }
        else if (y < tg67x)
        {
            int delta = ((x ^ y) < 0) ? -1 : 1;
            if (mag0 > mag[lidy + delta][lidx - 1] && mag0 > mag[lidy - delta][lidx + 1])
                canny_push(gidx, gidy)
        }
        else
        {
            if (mag0 > mag[lidy - 1][lidx] && mag0 > mag[lidy + 1][lidx])
                canny_push(gidx, gidy)
        }
    }
    
    storepix(value, map + mad24(gidy, map_step, mad24(gidx, sizeof(int), map_offset)));
}

#elif defined WITHOUT_SOBEL

/*
    stage1_without_sobel:
        Calc magnitudes
        Non maxima suppression
        Double thresholding
*/

#define loadpix(addr) (__global short *)(addr)
#define storepix(val, addr) *(__global int *)(addr) = (int)(val)

inline int dist(short x, short y)
{
#ifdef L2_GRAD
    return x * x + y * y;
#else
    return abs(x) + abs(y);
#endif
}

__kernel void stage1_without_sobel(__global const short *dxptr, int dx_step, int dx_offset,
                                   __global const short *dyptr, int dy_step, int dy_offset,
                                   __global uchar *map, int map_step, int map_offset,
                                   __global ushort2 *stack, __global int *counter,
                                   int low_thr, int high_thr)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    __local int mag[18][18];

    int dx_index = mad24(gidy, dx_step, mad24(gidx, sizeof(short) * cn, dx_offset));
    int dy_index = mad24(gidy, dy_step, mad24(gidx, sizeof(short) * cn, dy_offset));

    __global short *dx = loadpix(dxptr + dx_index);
    __global short *dy = loadpix(dyptr + dy_index);

    int mag0 = dist(dx[0], dy[0]);
#if cn > 1
    #pragma unroll
    short cdx = dx[0], cdy = dy[0];
    for (int i = 1; i < cn; ++i)
    {
        int mag1 = dist(dx[i], dy[i]); 
        if (mag1 > mag0)
        {
            mag0 = mag1;
            cdx = dx[i];
            cdy = dy[i]; 
        }
    }
    dx[0] = cdx;
    dy[0] = cdy;
#endif 
    mag[lidy][lidx] = mag0;

    barrier(CLK_LOCAL_MEM_FENCE);

    lidy = clamp(lidy, 1, 16);
    lidx = clamp(lidx, 1, 16);
    mag0 = mag[lidy][lidx];
    /*
        0 - might belong to an edge
        1 - pixel doesn't belong to an edge
        2 - belong to an edge
    */
    uchar value = 1;
    if (mag0 > low_thr)
    {
        int tg22x = dx[0] * TG22;
        int y = dy[0] << CANNY_SHIFT;
        int tg67x = tg22x + (dx[0] << (1 + CANNY_SHIFT));
        
        if (y < tg22x)
        {
            if (mag0 > mag[lidy][lidx - 1] && mag0 > mag[lidy][lidx + 1])
                canny_push(gidx, gidy);
        }
        else if(y < tg67x)
        {
            int delta = ((dx[0] ^ dy[0]) < 0) ? -1 : 1;
            if (mag0 > mag[lidy + delta][lidx - 1] && mag0 > mag[lidy - delta][lidx + 1])
                canny_push(gidx, gidy);
        }
        else
        {
            if (mag0 > mag[lidy - 1][lidx] && mag0 > mag[lidy + 1][lidx])
                canny_push(gidx, gidy);
        }
    }

    storepix(value, map + mad24(gidy, map_step, mad24(gidx, sizeof(int), map_offset)));
}

#undef TG22
#undef CANNY_SHIFT
#undef canny_push

#elif defined STAGE2
/*
    stage2:
        hysteresis (add edges labeled 1 if they are connected with an edge labeled 2)
*/
#define loadpix(addr) *(__global int *)(addr)
#define storepix(val, addr) *(__global int *)(addr) = (int)(val)
#define stack_size 256

__constant short move_dir[2][8] = {
    { -1, -1, -1, 0, 0, 1, 1, 1 },
    { -1, 0, 1, -1, 1, -1, 0, 1 }
};

__kernel void stage2_hysteresis(__global uchar *map, int map_step, int map_offset, int rows, int cols,
                                __global const ushort2 *common_stack)
{
    ushort2 stack[stack_size];
    uchar counter = 0;

    int gid = get_global_id(0);
    stack[0] = common_stack[gid];
    counter++;

    while (counter)
    {
        ushort2 pos = stack[counter - 1];
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            ushort posx = clamp(pos.x + move_dir[0][i], 0, cols - 1);
            ushort posy = clamp(pos.y + move_dir[1][i], 0, rows - 1);
            __global int *addr = map + mad24(posy, map_step, mad24(posx, sizeof(int), map_offset));
            int type = loadpix(addr);
            if (!type)
            {
                stack[counter] = (ushort2)(posx, posy);
                counter++;
                storepix(2, addr);
            }
        }
        counter--;
    }
}

#elif defined GET_EDGES

// Get the edge result. egde type of value 2 will be marked as an edge point and set to 255. Otherwise 0.
// map      edge type mappings
// dst      edge output

__kernel void getEdges(__global const uchar *mapptr, int map_step, int map_offset,
                       __global uchar *dst, int dst_step, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int map_index = mad24(map_step, y, mad24(x, sizeof(int), map_offset));
    int dst_index = mad24(dst_step, y, x) + dst_offset;

    __global const int * map = (__global const int *)(mapptr + map_index);

    dst[dst_index] = (uchar)(-(map[0] >> 1));
}

#endif