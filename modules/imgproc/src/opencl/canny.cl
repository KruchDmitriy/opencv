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
#define NEW_TG22 0.4142135623730950488016887242097f
#define NEW_TG67 2.4142135623730950488016887242097f


// how to write it more easier?
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
#define loadpix(addr) convert_intN(*(__global const TYPE *)(addr))
#else
#define loadpix(addr) convert_intN(vload3(0, (__global const TYPE *)(addr)))
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
    __local intN smem[GRP_SIZEY + 2][GRP_SIZEX + 2];

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    int gidx_im = get_global_id(0);
    int gidy_im = get_global_id(1);

    int gidx = gidx_im - (get_group_id(0) * 2 + 1);
    int gidy = gidy_im - (get_group_id(1) * 2 + 1);

    if (lidx == 0 && lidy == 0)
    {
        smem[0][0] = loadpix(src + mad24(max(gidy - 1, 0), src_step,
                                mad24(max(gidx - 1, 0), cn * sizeof(TYPE), src_offset)));
        smem[0][GRP_SIZEX + 1] = loadpix(src + mad24(max(gidy - 1, 0), src_step,
                                mad24(min(gidx + GRP_SIZEX, cols - 1), cn * sizeof(TYPE), src_offset)));
        smem[GRP_SIZEY + 1][0] = loadpix(src + mad24(min(gidy + GRP_SIZEY, rows - 1), src_step,
                                mad24(max(gidx - 1, 0), cn * sizeof(TYPE), src_offset)));
        smem[GRP_SIZEY + 1][GRP_SIZEX + 1] = loadpix(src + mad24(min(gidy + GRP_SIZEY, rows - 1), src_step,
                                mad24(min(gidx + GRP_SIZEX, cols - 1), cn * sizeof(TYPE), src_offset)));
    }
    if (lidx == 0)
    {
        smem[lidy + 1][0] = loadpix(src + mad24(clamp(gidy, 0, rows - 1), src_step,
                                mad24(max(gidx - 1, 0), cn * sizeof(TYPE), src_offset)));
        smem[lidy + 1][GRP_SIZEX + 1] = loadpix(src + mad24(clamp(gidy, 0, rows - 1), src_step,
                                mad24(min(gidx + GRP_SIZEX, cols - 1), cn * sizeof(TYPE), src_offset)));
    }
    if (lidy == 0)
    {
        smem[0][lidx + 1] = loadpix(src + mad24(max(gidy - 1, 0), src_step,
                                mad24(clamp(gidx, 0, cols - 1), cn * sizeof(TYPE), src_offset)));
        smem[GRP_SIZEY + 1][lidx + 1] = loadpix(src + mad24(min(gidy + GRP_SIZEY, rows - 1), src_step,
                                mad24(clamp(gidx, 0, cols - 1), cn * sizeof(TYPE), src_offset)));;
    }

    gidx = clamp(gidx, 0, cols - 1);
    gidy = clamp(gidy, 0, rows - 1);

    smem[lidy + 1][lidx + 1] = loadpix(src + mad24(gidy, src_step, mad24(gidx, cn * sizeof(TYPE), src_offset)));
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //// Sobel
    //

    int lx = lidx + 1;
    int ly = lidy + 1; 

    intN dx = smem[ly - 1][lx + 1] - smem[ly - 1][lx - 1]
            + 2 * (smem[ly][lx + 1] - smem[ly][lx - 1])
            + smem[ly + 1][lx + 1] - smem[ly + 1][lx - 1]; 

    intN dy = smem[ly - 1][lx - 1] - smem[ly + 1][lx - 1]
            + 2 * (smem[ly - 1][lx] - smem[ly + 1][lx])
            + smem[ly - 1][lx + 1] - smem[ly + 1][lx + 1];

    //// Magnitude
    //
    __local int mag[GRP_SIZEY][GRP_SIZEX];

    int x, y;
#ifdef L2GRAD
    intN magN = dx * dx + dy * dy;
#else
    intN magN = convert_intN(abs(dx) + abs(dy));
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

    /*
        A little bit magic

        sector numbers

        3   2   1
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        1   2   3

        We need to approximate arctg(dy / dx) to four direction 0, 45, 90 or 135 degrees.
        Therefore if abs(dy / dx) belongs to the interval
        [0, tg(22.5)]           -> 0 direction
        [tg(22.5), tg(67.5)]    -> 1 or 3
        [tg(67,5), +oo)         -> 2 
        
        Since tg(67.5) = 1 / tg(22.5), if we take
        a = abs(dy / dx) * tg(22.5) and b = abs(dy / dx) * tg(67.5)
        we can get another intervals
        
        in case a:
        [0, tg(22.5)^2]     -> 0
        [tg(22.5)^2, 1]     -> 1, 3
        [1, +oo)            -> 2

        in case b:
        [0, 1]              -> 0
        [1, tg(67.5)^2]     -> 1,3
        [tg(67.5)^2, +oo)   -> 2

        that can help to find direction without conditions. 

        0 - might belong to an edge
        1 - pixel doesn't belong to an edge
        2 - belong to an edge
    */

    __constant int prev[4][2] = {
        { 0, -1 },
        { -1, 1 },
        { -1, 0 },
        { -1, -1 }
    };

    __constant int next[4][2] = {
        { 0, 1 },
        { 1, -1 },
        { 1, 0 },         
        { 1, 1 }      
    };

    if (!(lidx > 0 && lidx < (GRP_SIZEX - 1) && lidy > 0 && lidy < (GRP_SIZEY - 1))) // needn't cols, rows
        return;

    int mag0 = mag[lidy][lidx];
    
    int value = 1;
    /*if (mag0 > low_thr)
    {
        int ax = abs(x);
        int ay = abs(y);
        int tg22x = ax * TG22;
        ay <<= CANNY_SHIFT;       

        if (ay < tg22x)
        {
            if (mag0 > mag[lidy][lidx - 1] && mag0 >= mag[lidy][lidx + 1])
            {
                canny_push(gidx, gidy)
            }
        }
        else 
        {
            int tg67x = tg22x + (ax << (1 + CANNY_SHIFT));
            if (ay > tg67x)
            {
                if (mag0 > mag[lidy - 1][lidx] && mag0 >= mag[lidy + 1][lidx])
                {
                    canny_push(gidx, gidy)
                }
            }
            else
            {
                int delta = ((x ^ y) < 0) ? 1 : -1;
                if (mag0 > mag[lidy - delta][lidx - 1] && mag0 > mag[lidy + delta][lidx + 1])
                {
                    canny_push(gidx, gidy)
                }
            }
        }
    }*/
    if (mag0 > low_thr)
    {
        int a = (y / (float)x) * NEW_TG22;
        int b = (y / (float)x) * NEW_TG67;

        int ai = min((int)abs(a), 1) + 1;
        int bi = min((int)abs(b), 1);

        //  ai = { 1, 2 }
        //  bi = { 0, 1 }
        //  ai * bi = { 0, 1, 2 } - directions that we need ( + 3 if x ^ y < 0)

        int dir3 = (ai * bi) & (((x ^ y) & 0x80000000) >> 31); // if ai = 1, bi = 1, dy ^ dx < 0
        int dir = ai * bi + 2 * dir3;
        int prev_mag = mag[lidy + prev[dir][0]][lidx + prev[dir][1]]; 
        int next_mag = mag[lidy + next[dir][0]][lidx + next[dir][1]] + (dir & 1);
        if (mag0 > prev_mag && mag0 >= next_mag)
        {
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

#ifdef L2GRAD
#define dist(x, y) ((int)(x) * (x) + (int)(y) * (y))
#else
#define dist(x, y) (abs(x) + abs(y))             
#endif


__kernel void stage1_without_sobel(__global const uchar *dxptr, int dx_step, int dx_offset, 
                                   __global const uchar *dyptr, int dy_step, int dy_offset,
                                   __global uchar *map, int map_step, int map_offset, int rows, int cols,
                                   __global ushort2 *stack, __global int *counter,
                                   int low_thr, int high_thr)
{
    int gidx_im = get_global_id(0);
    int gidy_im = get_global_id(1);

    int gidx = clamp((int)(gidx_im - (get_group_id(0) * 2 + 1)), 0, cols - 1);
    int gidy = clamp((int)(gidy_im - (get_group_id(1) * 2 + 1)), 0, rows - 1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    __local int mag[GRP_SIZEY][GRP_SIZEX];

    int dx_index = mad24(gidy, dx_step, mad24(gidx, cn * sizeof(short), dx_offset));
    int dy_index = mad24(gidy, dy_step, mad24(gidx, cn * sizeof(short), dy_offset));

    __global short *dx = loadpix(dxptr + dx_index);
    __global short *dy = loadpix(dyptr + dy_index);

    int mag0 = dist(dx[0], dy[0]);
#if cn > 1
    short cdx = dx[0], cdy = dy[0];
    #pragma unroll
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

    if (!(lidx > 0 && lidx < (GRP_SIZEX - 1) && lidy > 0 && lidy < (GRP_SIZEY - 1)))
        return;

    /*
        0 - might belong to an edge
        1 - pixel doesn't belong to an edge
        2 - belong to an edge
    */
    int value = 1;
    if (mag0 > low_thr)
    {
        int tg22x = ((int)abs(dx[0])) * TG22;
        int y = ((int)abs(dy[0])) << CANNY_SHIFT;
        int tg67x = tg22x + ((abs((int)dx[0])) << (1 + CANNY_SHIFT));
        
        if (y < tg22x)
        {
            if (mag0 > mag[lidy][lidx - 1] && mag0 >= mag[lidy][lidx + 1])
            {
                canny_push(gidx, gidy)
            }
        }
        else if(y > tg67x)
        {
            if (mag0 > mag[lidy - 1][lidx] && mag0 >= mag[lidy + 1][lidx])
            {
                canny_push(gidx, gidy)
            }    
        }
        else
        {
            int delta = ((dx[0] ^ dy[0]) < 0) ? -1 : 1;
            if (mag0 > mag[lidy - delta][lidx - 1] && mag0 > mag[lidy + delta][lidx + 1])
            {
                canny_push(gidx, gidy)
            }             
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
        hysteresis (add edges labeled 0 if they are connected with an edge labeled 2)
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
    __private ushort2 stack[stack_size];
    uchar counter = 0;

    int gid = get_global_id(0);
    stack[0] = common_stack[gid];
    counter++;

    while (counter != 0)
    {
        ushort2 pos = stack[--counter];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            ushort posx = clamp(pos.x + move_dir[0][i], 0, cols - 1);
            ushort posy = clamp(pos.y + move_dir[1][i], 0, rows - 1);
            __global int *addr = (__global int *)(map + mad24(posy, map_step, mad24(posx, sizeof(int), map_offset)));
            int type = loadpix(addr);
            if (type == 0)
            {
                stack[counter] = (ushort2)(posx, posy);
                counter++;
                storepix(2, addr);
            }
        }
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