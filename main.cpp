#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <SDL.h>
#include <SDL_metal.h>

#include <iostream>

int main(int argc, char** argv)
{
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    return 1;
  }

  const int width = 1024;
  const int height = 768;
  const float aspect = float(width) / float(height);
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

  // MTL::Device* device = MTLCreateSystemDefaultDevice();
  // CA::MetalLayer* metal_layer =
  // (CA::MetalLayer*)SDL_Metal_GetLayer(metal_view);

  // MTL::Device* device = metal_layer->device();
  SDL_Renderer* renderer =
    SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);

  // auto metal_layer = (CA::MetalLayer*)SDL_RenderGetMetalLayer(renderer);
  auto metal_layer = (CA::MetalLayer*)SDL_Metal_GetLayer(metal_view);
  // auto device = ml->device();
  MTL::Device* device = metal_layer->device();

  // if (metal_layer == ml) {
  //   std::cout << "same\n";
  // }

  auto name = device->name();
  std::cerr << "device name: " << name->utf8String() << std::endl;

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
      command_encoder->endEncoding();
      command_buffer->presentDrawable(current_drawable);
      command_buffer->commit();
    }

    pool->release();
  }

  // dummy
  // int data;
  // auto library_data =
  //   dispatch_data_create(&data, 4, dispatch_get_main_queue(), {});

  command_queue->release();
  device->release();

  return 0;
}
