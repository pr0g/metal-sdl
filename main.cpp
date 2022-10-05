#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <SDL.h>
#include <SDL_metal.h>

#include <iostream>

#include "vertex.h"

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
          uint vertex_id [[vertex_id]],
          constant VertexData* vertices [[buffer(0)]]) {
            rasterizer_data_t out;
            out.position = float4(vertices->pos_col[vertex_id].position.xy, 0.0, 1.0);
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

  MTL::RenderPipelineState* render_pipeline_state =
    device->newRenderPipelineState(pipeline_descriptor, &error);
  if (!render_pipeline_state) {
    std::cout << error->localizedDescription()->utf8String() << '\n';
    return 1;
  }

  pipeline_descriptor->release();
  library->release();

  const vertex_pos_col_t triangle_vertices[] = {
    {{-0.8f, -0.8f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.8f, -0.8f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
    {{0.0f, 0.8f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
  };

  MTL::Buffer* vert_pos_col_buffer = device->newBuffer(
    sizeof(triangle_vertices), MTL::ResourceStorageModeManaged);
  memcpy(
    vert_pos_col_buffer->contents(), triangle_vertices,
    sizeof(triangle_vertices));
  vert_pos_col_buffer->didModifyRange(
    NS::Range::Make(0, vert_pos_col_buffer->length()));

  MTL::ArgumentEncoder* arg_encoder = vert_fn->newArgumentEncoder(0);
  MTL::Buffer* arg_buffer = device->newBuffer(
    arg_encoder->encodedLength(), MTL::ResourceStorageModeManaged);
  arg_encoder->setArgumentBuffer(arg_buffer, 0);
  arg_encoder->setBuffer(vert_pos_col_buffer, 0, 0);
  arg_buffer->didModifyRange(NS::Range(0, arg_buffer->length()));

  vert_fn->release();
  frag_fn->release();
  arg_encoder->release();

  MTL::CommandQueue* command_queue = device->newCommandQueue();

  for (bool quit = false; !quit;) {
    for (SDL_Event current_event; SDL_PollEvent(&current_event) != 0;) {
      if (current_event.type == SDL_QUIT) {
        quit = true;
        break;
      }
    }

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    if (CA::MetalDrawable* current_drawable = metal_layer->nextDrawable()) {
      MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
      command_buffer->setLabel(NS::String::string(
        "SimpleCommand", NS::StringEncoding::UTF8StringEncoding));
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
      MTL::RenderCommandEncoder* command_encoder =
        command_buffer->renderCommandEncoder(pass_descriptor);
      command_encoder->setRenderPipelineState(render_pipeline_state);
      command_encoder->setViewport(
        MTL::Viewport{0, 0, width, height, 0.0, 1.0});
      command_encoder->setVertexBuffer(arg_buffer, 0, 0);
      command_encoder->useResource(vert_pos_col_buffer, MTL::ResourceUsageRead);
      command_encoder->drawPrimitives(
        MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(0),
        NS::UInteger(3));
      command_encoder->endEncoding();
      command_buffer->presentDrawable(current_drawable);
      command_buffer->commit();
    }

    pool->release();
  }

  command_queue->release();
  device->release();

  return 0;
}
