#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "imgui/imgui_impl_metal.h"
#include "imgui/imgui_impl_sdl.h"
#include "vertex.h"

#include <SDL.h>
#include <SDL_metal.h>

#include <as-camera-input-sdl/as-camera-input-sdl.hpp>
#include <as-camera-input/as-camera-input.hpp>
#include <as/as-view.hpp>

#include <chrono>
#include <iostream>

// standard linearization alternative
// return near * far / ((depth * (near - far)) + far);
// reverse linearization
// return near * far / ((depth * (far - near)) + near);

using fp_seconds = std::chrono::duration<float, std::chrono::seconds::period>;

enum class render_mode_e
{
  color,
  depth
};

enum class depth_mode_e
{
  normal,
  reverse
};

enum class layout_mode_e
{
  near,
  fighting
};

depth_mode_e g_depth_mode = depth_mode_e::normal;
render_mode_e g_render_mode = render_mode_e::color;
layout_mode_e g_layout_mode = layout_mode_e::near;

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

  MTL::TextureDescriptor* render_target_texture_desc =
    MTL::TextureDescriptor::alloc()->init();
  render_target_texture_desc->setTextureType(MTL::TextureType2D);
  render_target_texture_desc->setPixelFormat(
    MTL::PixelFormat::PixelFormatBGRA8Unorm);
  render_target_texture_desc->setWidth(width);
  render_target_texture_desc->setHeight(height);
  render_target_texture_desc->setUsage(
    MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget);
  MTL::Texture* render_target_texture =
    device->newTexture(render_target_texture_desc);
  render_target_texture_desc->release();

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

  MTL::RenderPassDescriptor* reverse_render_pass_desc_scene =
    MTL::RenderPassDescriptor::alloc()->init();
  reverse_render_pass_desc_scene->colorAttachments()->object(0)->setTexture(
    render_target_texture);
  reverse_render_pass_desc_scene->colorAttachments()->object(0)->setLoadAction(
    MTL::LoadActionClear);
  reverse_render_pass_desc_scene->colorAttachments()->object(0)->setStoreAction(
    MTL::StoreActionStore);
  reverse_render_pass_desc_scene->colorAttachments()->object(0)->setClearColor(
    MTL::ClearColor::Make(0.2, 0.3, 0.3, 1.0));
  reverse_render_pass_desc_scene->depthAttachment()->setClearDepth(0.0f);
  reverse_render_pass_desc_scene->depthAttachment()->setLoadAction(
    MTL::LoadActionClear);
  reverse_render_pass_desc_scene->depthAttachment()->setStoreAction(
    MTL::StoreActionStore);
  reverse_render_pass_desc_scene->depthAttachment()->setTexture(depth_texture);

  MTL::RenderPassDescriptor* normal_render_pass_desc_scene =
    MTL::RenderPassDescriptor::alloc()->init();
  normal_render_pass_desc_scene->colorAttachments()->object(0)->setTexture(
    render_target_texture);
  normal_render_pass_desc_scene->colorAttachments()->object(0)->setLoadAction(
    MTL::LoadActionClear);
  normal_render_pass_desc_scene->colorAttachments()->object(0)->setStoreAction(
    MTL::StoreActionStore);
  normal_render_pass_desc_scene->colorAttachments()->object(0)->setClearColor(
    MTL::ClearColor::Make(0.2, 0.3, 0.3, 1.0));
  normal_render_pass_desc_scene->depthAttachment()->setClearDepth(1.0f);
  normal_render_pass_desc_scene->depthAttachment()->setLoadAction(
    MTL::LoadActionClear);
  normal_render_pass_desc_scene->depthAttachment()->setStoreAction(
    MTL::StoreActionStore);
  normal_render_pass_desc_scene->depthAttachment()->setTexture(depth_texture);

  const char* shader_src_scene = R"(
        #include <metal_stdlib>
        #include "../../vertex.h"

        struct rasterizer_data_t {
          float4 position [[position]];
          float4 color;
        };

        struct VertexData {
            device vertex_pos_t* pos_col [[id(0)]];
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
            out.color = instance_data[instance_id].color;
            return out;
        }

        fragment float4 fragment_shader(rasterizer_data_t in [[stage_in]]) {
          return in.color;
        }
  )";

  using NS::StringEncoding::UTF8StringEncoding;
  NS::Error* error = nullptr;
  MTL::Library* library_scene = device->newLibrary(
    NS::String::string(shader_src_scene, UTF8StringEncoding), nullptr, &error);
  if (!library_scene) {
    std::cout << error->localizedDescription()->utf8String() << '\n';
    return 1;
  }

  MTL::Function* vert_fn_scene = library_scene->newFunction(
    NS::String::string("vertex_shader", UTF8StringEncoding));
  MTL::Function* frag_fn_scene = library_scene->newFunction(
    NS::String::string("fragment_shader", UTF8StringEncoding));

  MTL::RenderPipelineDescriptor* pipeline_descriptor_scene =
    MTL::RenderPipelineDescriptor::alloc()->init();
  pipeline_descriptor_scene->setLabel(
    NS::String::string("scene", UTF8StringEncoding));
  pipeline_descriptor_scene->setVertexFunction(vert_fn_scene);
  pipeline_descriptor_scene->setFragmentFunction(frag_fn_scene);
  pipeline_descriptor_scene->colorAttachments()->object(0)->setPixelFormat(
    MTL::PixelFormat::PixelFormatBGRA8Unorm);
  pipeline_descriptor_scene->setDepthAttachmentPixelFormat(
    MTL::PixelFormat::PixelFormatDepth32Float);
  pipeline_descriptor_scene->vertexBuffers()->object(0)->setMutability(
    MTL::MutabilityImmutable);

  MTL::RenderPipelineState* render_pipeline_state_scene =
    device->newRenderPipelineState(pipeline_descriptor_scene, &error);
  if (!render_pipeline_state_scene) {
    std::cout << error->localizedDescription()->utf8String() << '\n';
    return 1;
  }

  pipeline_descriptor_scene->release();

  MTL::RenderPipelineDescriptor* pipeline_descriptor_ui =
    MTL::RenderPipelineDescriptor::alloc()->init();
  pipeline_descriptor_ui->setLabel(
    NS::String::string("scene", UTF8StringEncoding));
  pipeline_descriptor_ui->setVertexFunction(vert_fn_scene);
  pipeline_descriptor_ui->setFragmentFunction(frag_fn_scene);
  pipeline_descriptor_ui->colorAttachments()->object(0)->setPixelFormat(
    MTL::PixelFormat::PixelFormatBGRA8Unorm);
  pipeline_descriptor_ui->setDepthAttachmentPixelFormat(
    MTL::PixelFormat::PixelFormatInvalid);
  pipeline_descriptor_ui->vertexBuffers()->object(0)->setMutability(
    MTL::MutabilityImmutable);

  MTL::RenderPipelineState* render_pipeline_state_ui =
    device->newRenderPipelineState(pipeline_descriptor_ui, &error);
  if (!render_pipeline_state_ui) {
    std::cout << error->localizedDescription()->utf8String() << '\n';
    return 1;
  }

  pipeline_descriptor_ui->release();

  library_scene->release();

  const char* shader_src_screen = R"(
        #include <metal_stdlib>
        #include "../../vertex.h"

        struct texture_rasterizer_data_t {
          float4 position [[position]];
          float2 texcoord;
        };

        vertex texture_rasterizer_data_t vertex_shader(
          constant vertex_pos_tex_t* vertices [[buffer(0)]],
          const uint vertex_id [[vertex_id]]) {
            texture_rasterizer_data_t out;
            out.position = simd::float4(vertices[vertex_id].position.xy, 0.0, 1.0);
            out.texcoord = vertices[vertex_id].texcoord;
            return out;
        }

        fragment float4 fragment_shader(
          texture_rasterizer_data_t in [[stage_in]],
          metal::texture2d<float> texture [[texture(0)]]) {
          metal::sampler simple_sampler;
          return texture.sample(simple_sampler, in.texcoord);
        }

        float linearize_depth(
          texture_rasterizer_data_t in [[stage_in]],
          metal::texture2d<float> texture [[texture(0)]],
          const float near, const float far)
        {
            metal::sampler simple_sampler;
            float depth = texture.sample(simple_sampler, in.texcoord).x;
            // inverse of perspective projection matrix transformation
            return (far * near) / ((depth * (near - far)) + far);
        }

        fragment float4 fragment_shader_depth(
          texture_rasterizer_data_t in [[stage_in]],
          constant frame_data_t* frame_data [[buffer(1)]],
          metal::texture2d<float> texture [[texture(0)]]) {
          const float near = frame_data->near;
          const float far  = frame_data->far;
          float c = linearize_depth(in, texture, near, far);
          float3 range = float3(c - near) / (far - near); // convert to [0,1]
          return float4(range, 1.0);
        }
  )";

  using NS::StringEncoding::UTF8StringEncoding;
  MTL::Library* library_screen = device->newLibrary(
    NS::String::string(shader_src_screen, UTF8StringEncoding), nullptr, &error);
  if (!library_screen) {
    std::cout << error->localizedDescription()->utf8String() << '\n';
    return 1;
  }

  MTL::Function* vert_fn_screen = library_screen->newFunction(
    NS::String::string("vertex_shader", UTF8StringEncoding));
  MTL::Function* frag_fn_screen = library_screen->newFunction(
    NS::String::string("fragment_shader", UTF8StringEncoding));
  MTL::Function* frag_fn_screen_depth = library_screen->newFunction(
    NS::String::string("fragment_shader_depth", UTF8StringEncoding));

  MTL::RenderPipelineDescriptor* pipeline_descriptor_screen =
    MTL::RenderPipelineDescriptor::alloc()->init();
  pipeline_descriptor_screen->setLabel(
    NS::String::string("screen", UTF8StringEncoding));
  pipeline_descriptor_screen->setVertexFunction(vert_fn_screen);
  pipeline_descriptor_screen->setFragmentFunction(frag_fn_screen);
  pipeline_descriptor_screen->colorAttachments()->object(0)->setPixelFormat(
    render_target_texture->pixelFormat());

  MTL::RenderPipelineState* render_pipeline_state_screen =
    device->newRenderPipelineState(pipeline_descriptor_screen, &error);
  if (!render_pipeline_state_screen) {
    std::cout << error->localizedDescription()->utf8String() << '\n';
    return 1;
  }

  pipeline_descriptor_screen->release();

  MTL::RenderPipelineDescriptor* pipeline_descriptor_screen_depth =
    MTL::RenderPipelineDescriptor::alloc()->init();
  pipeline_descriptor_screen_depth->setLabel(
    NS::String::string("screen-depth", UTF8StringEncoding));
  pipeline_descriptor_screen_depth->setVertexFunction(vert_fn_screen);
  pipeline_descriptor_screen_depth->setFragmentFunction(frag_fn_screen_depth);
  pipeline_descriptor_screen_depth->colorAttachments()
    ->object(0)
    ->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm);

  MTL::RenderPipelineState* render_pipeline_state_screen_depth =
    device->newRenderPipelineState(pipeline_descriptor_screen_depth, &error);
  if (!render_pipeline_state_screen_depth) {
    std::cout << error->localizedDescription()->utf8String() << '\n';
    return 1;
  }

  pipeline_descriptor_screen_depth->release();
  library_screen->release();

  const uint16_t indices[] = {0, 1, 2, 0, 2, 3};
  const vertex_pos_t vertices[] = {
    {{-0.5f, -0.5f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}},
    {{-0.5f, 0.5f, 0.0f}}};

  MTL::Buffer* vertex_buffer =
    device->newBuffer(sizeof(vertices), MTL::ResourceStorageModeManaged);
  memcpy(vertex_buffer->contents(), vertices, sizeof(vertices));
  vertex_buffer->didModifyRange(NS::Range::Make(0, vertex_buffer->length()));

  MTL::Buffer* index_buffer =
    device->newBuffer(sizeof(indices), MTL::ResourceStorageModeManaged);
  memcpy(index_buffer->contents(), indices, sizeof(indices));
  index_buffer->didModifyRange(NS::Range::Make(0, index_buffer->length()));

  MTL::ArgumentEncoder* arg_encoder = vert_fn_scene->newArgumentEncoder(0);
  MTL::Buffer* arg_buffer = device->newBuffer(
    arg_encoder->encodedLength(), MTL::ResourceStorageModeManaged);
  arg_encoder->setArgumentBuffer(arg_buffer, 0);
  arg_encoder->setBuffer(vertex_buffer, 0, 0);
  arg_buffer->didModifyRange(NS::Range(0, arg_buffer->length()));

  vert_fn_scene->release();
  frag_fn_scene->release();
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

  MTL::DepthStencilDescriptor* reverse_depth_stencil_desc =
    MTL::DepthStencilDescriptor::alloc()->init();
  reverse_depth_stencil_desc->setDepthCompareFunction(
    MTL::CompareFunction::CompareFunctionGreater);
  reverse_depth_stencil_desc->setDepthWriteEnabled(true);
  MTL::DepthStencilState* reverse_depth_stencil_state =
    device->newDepthStencilState(reverse_depth_stencil_desc);
  reverse_depth_stencil_desc->release();

  MTL::DepthStencilDescriptor* normal_depth_stencil_desc =
    MTL::DepthStencilDescriptor::alloc()->init();
  normal_depth_stencil_desc->setDepthCompareFunction(
    MTL::CompareFunction::CompareFunctionLess);
  normal_depth_stencil_desc->setDepthWriteEnabled(true);
  MTL::DepthStencilState* normal_depth_stencil_state =
    device->newDepthStencilState(normal_depth_stencil_desc);
  normal_depth_stencil_desc->release();

  dispatch_semaphore_t semaphore = dispatch_semaphore_create(MaxFramesInFlight);

  asc::Camera camera;
  camera.pivot = as::vec3(0.0f, 0.0f, -4.0f);
  asc::Camera target_camera = camera;

  asci::CameraSystem camera_system;
  asci::TranslateCameraInput translate_camera{
    asci::lookTranslation, asci::translatePivot};
  asci::RotateCameraInput rotate_camera{asci::MouseButton::Right};
  camera_system.cameras_.addCamera(&translate_camera);
  camera_system.cameras_.addCamera(&rotate_camera);

  float near = 5.0f;
  float far = 100.0f;

  ImGui::CreateContext();

  ImGui_ImplSDL2_InitForMetal(window);
  ImGui_ImplMetal_Init(device);

  MTL::CommandQueue* command_queue = device->newCommandQueue();

  layout_mode_e prev_layout_mode = g_layout_mode;
  auto prev = std::chrono::system_clock::now();
  for (bool quit = false; !quit;) {
    for (SDL_Event current_event; SDL_PollEvent(&current_event) != 0;) {
      ImGui_ImplSDL2_ProcessEvent(&current_event);
      if (current_event.type == SDL_QUIT) {
        quit = true;
        break;
      }
      camera_system.handleEvents(asci_sdl::sdlToInput(&current_event));
    }

    auto now = std::chrono::system_clock::now();
    auto delta = now - prev;
    prev = now;

    const float delta_time =
      std::chrono::duration_cast<fp_seconds>(delta).count();

    target_camera = camera_system.stepCamera(target_camera, delta_time);
    camera = asci::smoothCamera(
      camera, target_camera, asci::SmoothProps{}, delta_time);

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    frame_index = (frame_index + 1) % MaxFramesInFlight;
    MTL::Buffer* frame_data_buffer = frame_data_buffers[frame_index];
    MTL::Buffer* instance_data_buffer = instance_data_buffers[frame_index];

    if (CA::MetalDrawable* current_drawable = metal_layer->nextDrawable()) {
      MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
      command_buffer->setLabel(NS::String::string(
        "Command Buffer", NS::StringEncoding::UTF8StringEncoding));
      dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
      command_buffer->addCompletedHandler([&semaphore](MTL::CommandBuffer*) {
        dispatch_semaphore_signal(semaphore);
      });

      const as::mat4 perspective_projection = as::perspective_metal_lh(
        as::radians(60.0f), float(width) / float(height), near, far);
      const as::mat4 reverse_z_perspective_projection =
        as::reverse_z(perspective_projection);

      if (g_layout_mode != prev_layout_mode) {
        if (g_layout_mode == layout_mode_e::fighting) {
          near = 0.01f;
          far = 10000.0f;
        } else if (g_layout_mode == layout_mode_e::near) {
          near = 5.0f;
          far = 100.0f;
        }
        prev_layout_mode = g_layout_mode;
      }

      auto* frame_data =
        static_cast<frame_data_t*>(frame_data_buffer->contents());

      const as::mat4 view_projection = [camera, perspective_projection,
                                        reverse_z_perspective_projection] {
        const as::mat4 view = as::mat4_from_affine(camera.view());
        if (g_depth_mode == depth_mode_e::normal) {
          return as::mat_mul(view, perspective_projection);
        }
        if (g_depth_mode == depth_mode_e::reverse) {
          return as::mat_mul(view, reverse_z_perspective_projection);
        }
        return as::mat4::identity();
      }();

      const auto& vp = view_projection;
      frame_data->view_projection = simd::float4x4{
        simd::float4{vp[0], vp[1], vp[2], vp[3]},
        simd::float4{vp[4], vp[5], vp[6], vp[7]},
        simd::float4{vp[8], vp[9], vp[10], vp[11]},
        simd::float4{vp[12], vp[13], vp[14], vp[15]}};
      frame_data->near = near;
      frame_data->far = far;
      frame_data_buffer->didModifyRange(
        NS::Range::Make(0, sizeof(frame_data_t)));

      const as::affine near_positions[InstanceCount] = {
        as::affine_from_vec3(as::vec3{-0.25f, 0.25f, 1.0f}),
        as::affine_from_vec3(as::vec3{0.25f, -0.25f, 3.0f}),
        as::affine_from_vec3(as::vec3{-0.25f, 5.5f, 20.0f}),
        as::affine_from_vec3(as::vec3{-30.0f, 0.0f, 80.0f})};
      const as::affine fighting_positions[InstanceCount] = {
        as::affine_from_mat3_vec3(
          as::mat3_scale(100.0f, 100.0f, 1.0f),
          as::vec3{-10.0f, 25.0f, 500.02f}),
        as::affine_from_mat3_vec3(
          as::mat3_scale(100.0f, 100.0f, 1.0f),
          as::vec3{10.0f, -25.0f, 499.98f}),
        as::affine_from_mat3_vec3(
          as::mat3_scale(100.0f, 100.0f, 1.0f), as::vec3{-10.0f, 0.0f, 500.0f}),
        as::affine_from_mat3_vec3(
          as::mat3_scale(100.0f, 100.0f, 1.0f),
          as::vec3{10.0f, 0.0f, 500.01f})};
      const as::affine* transforms = g_layout_mode == layout_mode_e::near
                                     ? near_positions
                                     : fighting_positions;

      const as::vec4 colors[InstanceCount] = {
        as::vec4(1.0f, 0.5f, 0.2f, 1.0f), as::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        as::vec4(0.1f, 0.2f, 0.6f, 1.0f), as::vec4(0.1f, 0.8f, 0.2f, 1.0f)};

      auto* instance_data =
        static_cast<instance_data_t*>(instance_data_buffer->contents());
      for (int64_t i = 0; i < InstanceCount; ++i) {
        instance_data[i].model = simd::float4x4{
          simd::float4{
            transforms[i].rotation[0], transforms[i].rotation[1],
            transforms[i].rotation[2], 0.0f},
          simd::float4{
            transforms[i].rotation[3], transforms[i].rotation[4],
            transforms[i].rotation[5], 0.0f},
          simd::float4{
            transforms[i].rotation[6], transforms[i].rotation[7],
            transforms[i].rotation[8], 0.0f},
          simd::float4{
            transforms[i].translation.x, transforms[i].translation.y,
            transforms[i].translation.z, 1.0f}};
        instance_data[i].color =
          simd::float4{colors[i].x, colors[i].y, colors[i].z, colors[i].w};
      }

      instance_data_buffer->didModifyRange(
        NS::Range::Make(0, instance_data_buffer->length()));

      if (
        MTL::RenderCommandEncoder* render_command_encoder =
          command_buffer->renderCommandEncoder(
            g_depth_mode == depth_mode_e::reverse
              ? reverse_render_pass_desc_scene
              : normal_render_pass_desc_scene)) {
        render_command_encoder->setLabel(
          NS::String::string("Scene Pass", UTF8StringEncoding));
        render_command_encoder->setRenderPipelineState(
          render_pipeline_state_scene);
        render_command_encoder->setDepthStencilState(
          g_depth_mode == depth_mode_e::reverse ? reverse_depth_stencil_state
                                                : normal_depth_stencil_state);
        render_command_encoder->setCullMode(MTL::CullModeBack);
        render_command_encoder->setFrontFacingWinding(
          MTL::Winding::WindingCounterClockwise);
        render_command_encoder->setViewport(
          MTL::Viewport{0, 0, width, height, 0.0, 1.0});
        render_command_encoder->setVertexBuffer(arg_buffer, 0, 0);
        render_command_encoder->useResource(
          vertex_buffer, MTL::ResourceUsageRead);
        render_command_encoder->setVertexBuffer(frame_data_buffer, 0, 1);
        render_command_encoder->setVertexBuffer(instance_data_buffer, 0, 2);
        render_command_encoder->drawIndexedPrimitives(
          MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(6),
          MTL::IndexType::IndexTypeUInt16, index_buffer, NS::UInteger(0),
          InstanceCount);

        render_command_encoder->endEncoding();
      }

      MTL::RenderPassDescriptor* render_pass_desc_screen =
        MTL::RenderPassDescriptor::renderPassDescriptor();
      render_pass_desc_screen->colorAttachments()->object(0)->setLoadAction(
        MTL::LoadActionClear);
      render_pass_desc_screen->colorAttachments()->object(0)->setStoreAction(
        MTL::StoreActionStore);
      render_pass_desc_screen->colorAttachments()->object(0)->setTexture(
        current_drawable->texture());

      static const vertex_pos_tex_t quad_vertices[] = {
        {{1.0f, -1.0f}, {1.0f, 1.0f}}, {{-1.0f, -1.0f}, {0.0f, 1.0f}},
        {{-1.0f, 1.0f}, {0.0f, 0.0f}}, {{1.0f, -1.0f}, {1.0f, 1.0f}},
        {{-1.0f, 1.0f}, {0.0f, 0.0f}}, {{1.0f, 1.0f}, {1.0f, 0.0f}},
      };

      if (
        MTL::RenderCommandEncoder* render_command_encoder =
          command_buffer->renderCommandEncoder(render_pass_desc_screen)) {
        render_command_encoder->setLabel(
          NS::String::string("Screen Pass", UTF8StringEncoding));
        render_command_encoder->setViewport(
          MTL::Viewport{0, 0, width, height, 0.0, 1.0});
        render_command_encoder->setRenderPipelineState(
          g_render_mode == render_mode_e::color
            ? render_pipeline_state_screen
            : render_pipeline_state_screen_depth);
        render_command_encoder->setVertexBytes(
          &quad_vertices, sizeof(quad_vertices), 0);
        if (g_render_mode == render_mode_e::depth) {
          render_command_encoder->setFragmentBuffer(frame_data_buffer, 0, 1);
        }
        render_command_encoder->setFragmentTexture(
          g_render_mode == render_mode_e::color ? render_target_texture
                                                : depth_texture,
          0);
        render_command_encoder->drawPrimitives(
          MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(6));
        render_command_encoder->endEncoding();
      }

      MTL::RenderPassDescriptor* render_pass_desc_ui =
        MTL::RenderPassDescriptor::alloc()->init();
      render_pass_desc_ui->colorAttachments()->object(0)->setLoadAction(
        MTL::LoadActionLoad);
      render_pass_desc_ui->colorAttachments()->object(0)->setStoreAction(
        MTL::StoreActionStore);
      render_pass_desc_ui->colorAttachments()->object(0)->setTexture(
        current_drawable->texture());

      if (
        MTL::RenderCommandEncoder* render_command_encoder =
          command_buffer->renderCommandEncoder(render_pass_desc_ui)) {
        render_command_encoder->setLabel(
          NS::String::string("UI Pass", UTF8StringEncoding));
        render_command_encoder->setRenderPipelineState(
          render_pipeline_state_ui);
        render_command_encoder->setCullMode(MTL::CullModeBack);
        render_command_encoder->setFrontFacingWinding(
          MTL::Winding::WindingCounterClockwise);
        render_command_encoder->setViewport(
          MTL::Viewport{0, 0, width, height, 0.0, 1.0});

        ImGui_ImplMetal_NewFrame(render_pass_desc_ui);
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        {
          int depth_mode_index = static_cast<int>(g_depth_mode);
          const char* depth_mode_names[] = {"Normal", "Reverse"};
          ImGui::Combo(
            "Depth Mode", &depth_mode_index, depth_mode_names,
            std::size(depth_mode_names));
          g_depth_mode = static_cast<depth_mode_e>(depth_mode_index);
        }

        {
          int render_mode_index = static_cast<int>(g_render_mode);
          const char* render_mode_names[] = {"Color", "Depth"};
          ImGui::Combo(
            "Render Mode", &render_mode_index, render_mode_names,
            std::size(render_mode_names));
          g_render_mode = static_cast<render_mode_e>(render_mode_index);
        }

        {
          int layout_mode_index = static_cast<int>(g_layout_mode);
          const char* layout_mode_names[] = {"Near", "Fighting"};
          ImGui::Combo(
            "Layout Mode", &layout_mode_index, layout_mode_names,
            std::size(layout_mode_names));
          g_layout_mode = static_cast<layout_mode_e>(layout_mode_index);
        }

        ImGui::SliderFloat("Near Plane", &near, 0.01f, 49.9f);
        ImGui::SliderFloat("Far Plane", &far, 50.0f, 10000.0f);

        ImGui::Render();
        ImGui_ImplMetal_RenderDrawData(
          ImGui::GetDrawData(), command_buffer, render_command_encoder);

        render_command_encoder->endEncoding();
      }

      command_buffer->presentDrawable(current_drawable);
      command_buffer->commit();
    }

    pool->release();
  }

  ImGui_ImplMetal_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  reverse_render_pass_desc_scene->release();
  normal_render_pass_desc_scene->release();
  render_target_texture->release();
  depth_texture->release();
  arg_buffer->release();
  vertex_buffer->release();
  index_buffer->release();
  reverse_depth_stencil_state->release();
  normal_depth_stencil_state->release();
  for (int i = 0; i < MaxFramesInFlight; ++i) {
    frame_data_buffers[i]->release();
    instance_data_buffers[i]->release();
  }
  command_queue->release();
  device->release();

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
