#ifndef METAL_VERTEX_H
#define METAL_VERTEX_H

#include <simd/vector_types.h>

struct vertex_pos_col_t
{
  simd::float3 position;
  simd::float4 color;
};

#endif // METAL_VERTEX_H
