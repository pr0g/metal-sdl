#ifndef METAL_VERTEX_H
#define METAL_VERTEX_H

#include <simd/vector_types.h>
#include <simd/matrix_types.h>

struct vertex_pos_col_t
{
  simd::float3 position;
  simd::float4 color;
};

struct vertex_pos_tex_t
{
  simd::float2 position;
  simd::float2 texcoord;
};

struct frame_data_t
{
  simd::float4x4 view_projection;
  float near;
  float far;
};

struct instance_data_t
{
  simd::float4x4 model;
  simd::float4 color;
};

#endif // METAL_VERTEX_H
