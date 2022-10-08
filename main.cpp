#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <SDL.h>
#include <SDL_metal.h>

#include <as-camera-input/as-camera-input.hpp>
#include <as/as-view.hpp>

#include <iostream>

#include "vertex.h"

namespace asc
{
Handedness handedness()
{
  return Handedness::Left;
}
} // namespace asc

int main(int argc, char** argv)
{
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    return 1;
  }

  const int width = 1024;
  const int height = 768;
  SDL_Window* window = SDL_CreateWindow(
    argv[0], SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height,
    SDL_WINDOW_SHOWN | SDL_WINDOW_METAL);

  if (window == nullptr) {
    printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
    return 1;
  }

  SDL_MetalView metal_view = SDL_Metal_CreateView(window);
  if (metal_view == nullptr) {
    printf(
      "SDL_MetalView could not be created! SDL_Error: %s\n", SDL_GetError());
    return 1;
  }

  [[maybe_unused]] SDL_Renderer* renderer =
    SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);

  auto metal_layer = (CA::MetalLayer*)SDL_Metal_GetLayer(metal_view);
  metal_layer->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm);
  MTL::Device* device = metal_layer->device();

  auto name = device->name();
  std::cerr << "device name: " << name->utf8String() << std::endl;

  const char* shader_src = R"(
        #include <metal_stdlib>
        #include "../../vertex.h"

        struct rasterizer_data_t {
          float4 position [[position]];
          float4 color;
        };

        struct VertexData {
            device vertex_pos_col_t* pos_col [[id(0)]];
        };

        vertex rasterizer_data_t vertex_shader(
          constant VertexData* vertices [[buffer(0)]],
          constant frame_data_t* frame_data [[buffer(1)]],
          constant instance_data_t* instance_data [[buffer(2)]],
          uint vertex_id [[vertex_id]],
          uint instance_id [[instance_id]]) {
            rasterizer_data_t out;
            out.position =
              frame_data->view_projection
              * instance_data[instance_id].model
              * float4(vertices->pos_col[vertex_id].position.xy, 0.0, 1.0);
            out.color = vertices->pos_col[vertex_id].color;
            return out;
        }

        fragment float4 fragment_shader(rasterizer_data_t in [[stage_in]]) {
          return in.color;
        }
  )";

  using NS::StringEncoding::UTF8StringEncoding;
  NS::Error* error = nullptr;
  MTL::Library* library = device->newLibrary(
    NS::String::string(shader_src, UTF8StringEncoding), nullptr, &error);
  if (!library) {
    std::cout << error->localizedDescription()->utf8String() << '\n';
    return 1;
  }

  MTL::Function* vert_fn = library->newFunction(
    NS::String::string("vertex_shader", UTF8StringEncoding));
  MTL::Function* frag_fn = library->newFunction(
    NS::String::string("fragment_shader", UTF8StringEncoding));

  MTL::RenderPipelineDescriptor* pipeline_descriptor =
    MTL::RenderPipelineDescriptor::alloc()->init();
  pipeline_descriptor->setVertexFunction(vert_fn);
  pipeline_descriptor->setFragmentFunction(frag_fn);
  pipeline_descriptor->colorAttachments()->object(0)->setPixelFormat(
    MTL::PixelFormat::PixelFormatBGRA8Unorm);
  pipeline_descriptor->setDepthAttachmentPixelFormat(
    MTL::PixelFormat::PixelFormatDepth32Float);

  MTL::RenderPipelineState* render_pipeline_state =
    device->newRenderPipelineState(pipeline_descriptor, &error);
  if (!render_pipeline_state) {
    std::cout << error->localizedDescription()->utf8String() << '\n';
    return 1;
  }

  pipeline_descriptor->release();
  library->release();

  const uint16_t indices[] = {0, 1, 2, 0, 2, 3};
  const vertex_pos_col_t vertices[] = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}}};

  MTL::Buffer* vertex_buffer =
    device->newBuffer(sizeof(vertices), MTL::ResourceStorageModeManaged);
  memcpy(vertex_buffer->contents(), vertices, sizeof(vertices));
  vertex_buffer->didModifyRange(NS::Range::Make(0, vertex_buffer->length()));

  MTL::Buffer* index_buffer =
    device->newBuffer(sizeof(indices), MTL::ResourceStorageModeManaged);
  memcpy(index_buffer->contents(), indices, sizeof(indices));
  index_buffer->didModifyRange(NS::Range::Make(0, index_buffer->length()));

  MTL::ArgumentEncoder* arg_encoder = vert_fn->newArgumentEncoder(0);
  MTL::Buffer* arg_buffer = device->newBuffer(
    arg_encoder->encodedLength(), MTL::ResourceStorageModeManaged);
  arg_encoder->setArgumentBuffer(arg_buffer, 0);
  arg_encoder->setBuffer(vertex_buffer, 0, 0);
  arg_buffer->didModifyRange(NS::Range(0, arg_buffer->length()));

  vert_fn->release();
  frag_fn->release();
  arg_encoder->release();

  const int32_t MaxFramesInFlight = 3;
  const int32_t InstanceCount = 4;

  // instance data
  const int32_t instance_data_size =
    MaxFramesInFlight * InstanceCount * sizeof(instance_data_t);
  MTL::Buffer* instance_data_buffers[MaxFramesInFlight] = {};
  for (int i = 0; i < MaxFramesInFlight; ++i) {
    instance_data_buffers[i] =
      device->newBuffer(instance_data_size, MTL::ResourceStorageModeManaged);
  }

  // uniform data
  int frame_index = 0;
  MTL::Buffer* frame_data_buffers[MaxFramesInFlight] = {};
  for (int i = 0; i < MaxFramesInFlight; ++i) {
    frame_data_buffers[i] =
      device->newBuffer(sizeof(frame_data_t), MTL::ResourceStorageModeManaged);
  }

  MTL::DepthStencilDescriptor* depth_stencil_desc =
    MTL::DepthStencilDescriptor::alloc()->init();
  depth_stencil_desc->setDepthCompareFunction(
    MTL::CompareFunction::CompareFunctionLess);
  depth_stencil_desc->setDepthWriteEnabled(true);

  MTL::DepthStencilState* depth_stencil_state =
    device->newDepthStencilState(depth_stencil_desc);

  depth_stencil_desc->release();

  MTL::TextureDescriptor* depth_texture_desc =
    MTL::TextureDescriptor::alloc()->init();
  depth_texture_desc->setTextureType(MTL::TextureType2D);
  depth_texture_desc->setPixelFormat(MTL::PixelFormat::PixelFormatDepth32Float);
  depth_texture_desc->setWidth(width);
  depth_texture_desc->setHeight(height);
  depth_texture_desc->setStorageMode(MTL::StorageModeManaged);
  depth_texture_desc->setUsage(
    MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget);

  MTL::Texture* depth_texture = device->newTexture(depth_texture_desc);

  depth_texture_desc->release();

  dispatch_semaphore_t semaphore = dispatch_semaphore_create(MaxFramesInFlight);

  asc::Camera camera;
  camera.pivot = as::vec3(0.0f, 0.0f, -2.0f);
  const as::mat4 perspective_projection = as::perspective_d3d_lh(
    as::radians(60.0f), float(width) / float(height), 0.1f, 100.0f);

  MTL::CommandQueue* command_queue = device->newCommandQueue();
  for (bool quit = false; !quit;) {
    for (SDL_Event current_event; SDL_PollEvent(&current_event) != 0;) {
      if (current_event.type == SDL_QUIT) {
        quit = true;
        break;
      }
      if (current_event.type == SDL_KEYDOWN) {
        const auto* keyboard_event = (SDL_KeyboardEvent*)&current_event;
        if (keyboard_event->keysym.scancode == SDL_SCANCODE_S) {
          camera.pivot -= as::vec3::axis_z(0.1f);
        }
        if (keyboard_event->keysym.scancode == SDL_SCANCODE_W) {
          camera.pivot += as::vec3::axis_z(0.1f);
        }
      }
    }

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    frame_index = (frame_index + 1) % MaxFramesInFlight;
    MTL::Buffer* frame_data_buffer = frame_data_buffers[frame_index];
    MTL::Buffer* instance_data_buffer = instance_data_buffers[frame_index];

    if (CA::MetalDrawable* current_drawable = metal_layer->nextDrawable()) {
      MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
      command_buffer->setLabel(NS::String::string(
        "SimpleCommand", NS::StringEncoding::UTF8StringEncoding));
      dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
      command_buffer->addCompletedHandler([&semaphore](MTL::CommandBuffer*) {
        dispatch_semaphore_signal(semaphore);
      });

      auto* frame_data =
        static_cast<frame_data_t*>(frame_data_buffer->contents());
      const as::mat4 view = as::mat4_from_affine(camera.view());
      const as::mat4 view_projection = perspective_projection * view;
      const auto& vp = view_projection;
      frame_data->view_projection = simd::float4x4{
        simd::float4{vp[0], vp[1], vp[2], vp[3]},
        simd::float4{vp[4], vp[5], vp[6], vp[7]},
        simd::float4{vp[8], vp[9], vp[10], vp[11]},
        simd::float4{vp[12], vp[13], vp[14], vp[15]}};
      frame_data_buffer->didModifyRange(
        NS::Range::Make(0, sizeof(frame_data_t)));

      const as::vec3 positions[InstanceCount] = {
        as::vec3{-0.25f, 0.25f, 1.0f}, as::vec3{0.25f, -0.25f, 3.0f},
        as::vec3{-0.25f, 5.5f, 20.0f}, as::vec3{-30.0f, 0.0f, 80.0f}};

      auto* instance_data =
        static_cast<instance_data_t*>(instance_data_buffer->contents());
      for (int64_t i = 0; i < InstanceCount; ++i) {
        instance_data[i].model = simd::float4x4{
          simd::float4{1.0f, 0.0f, 0.0f, 0.0f},
          simd::float4{0.0f, 1.0f, 0.0f, 0.0f},
          simd::float4{0.0f, 0.0f, 1.0f, 0.0f},
          simd::float4{positions[i].x, positions[i].y, positions[i].z, 1.0f}};
      }

      instance_data_buffer->didModifyRange(
        NS::Range::Make(0, instance_data_buffer->length()));

      MTL::RenderPassDescriptor* pass_descriptor =
        MTL::RenderPassDescriptor::renderPassDescriptor();
      pass_descriptor->colorAttachments()->object(0)->setLoadAction(
        MTL::LoadActionClear);
      pass_descriptor->colorAttachments()->object(0)->setStoreAction(
        MTL::StoreActionStore);
      pass_descriptor->colorAttachments()->object(0)->setClearColor(
        MTL::ClearColor::Make(0.3922, 0.5843, 0.9294, 1.0));
      pass_descriptor->colorAttachments()->object(0)->setTexture(
        current_drawable->texture());
      pass_descriptor->depthAttachment()->setClearDepth(1.0f);
      pass_descriptor->depthAttachment()->setLoadAction(MTL::LoadActionClear);
      pass_descriptor->depthAttachment()->setStoreAction(MTL::StoreActionStore);
      pass_descriptor->depthAttachment()->setTexture(depth_texture);
      MTL::RenderCommandEncoder* command_encoder =
        command_buffer->renderCommandEncoder(pass_descriptor);
      command_encoder->setRenderPipelineState(render_pipeline_state);
      command_encoder->setDepthStencilState(depth_stencil_state);
      command_encoder->setCullMode(MTL::CullModeBack);
      command_encoder->setFrontFacingWinding(
        MTL::Winding::WindingCounterClockwise);
      command_encoder->setViewport(
        MTL::Viewport{0, 0, width, height, 0.0, 1.0});
      command_encoder->setVertexBuffer(arg_buffer, 0, 0);
      command_encoder->useResource(vertex_buffer, MTL::ResourceUsageRead);
      command_encoder->setVertexBuffer(frame_data_buffer, 0, 1);
      command_encoder->setVertexBuffer(instance_data_buffer, 0, 2);
      command_encoder->drawIndexedPrimitives(
        MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(6),
        MTL::IndexType::IndexTypeUInt16, index_buffer, NS::UInteger(0),
        InstanceCount);
      command_encoder->endEncoding();
      command_buffer->presentDrawable(current_drawable);
      command_buffer->commit();
    }

    pool->release();
  }

  depth_texture->release();
  arg_buffer->release();
  vertex_buffer->release();
  index_buffer->release();
  depth_stencil_state->release();
  for (int i = 0; i < MaxFramesInFlight; ++i) {
    frame_data_buffers[i]->release();
    instance_data_buffers[i]->release();
  }
  command_queue->release();
  device->release();

  return 0;
}
