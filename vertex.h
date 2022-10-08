#ifndef METAL_VERTEX_H
#define METAL_VERTEX_H

#include <simd/vector_types.h>
#include <simd/matrix_types.h>

struct vertex_pos_col_t
{
  simd::float3 position;
  simd::float4 color;
};

struct frame_data_t
{
  simd::float4x4 mvp;
};

#endif // METAL_VERTEX_H
