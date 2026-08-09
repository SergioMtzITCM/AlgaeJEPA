/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "esp_heap_caps.h"

#include "main_functions.h"
#include "model.h"
#include "constants.h"
#include "output_handler.h"
#include "test_data.h"

// Latency & RAM Metrics
#include "esp_timer.h"      // esp_timer_get_time() -> microsegundos desde el arranque
#include "esp_system.h"     // esp_get_free_heap_size(), esp_get_minimum_free_heap_size()
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"  // uxTaskGetStackHighWaterMark()

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;


constexpr int kTensorArenaSize = 3 * 1024 * 1024; // 3 MB
uint8_t* tensor_arena = nullptr;

bool inference_done = false;

bool model_ready = false;

// Latency variables
int64_t latency_min_us = INT64_MAX;
int64_t latency_max_us = 0;
int64_t latency_sum_us = 0;
uint32_t latency_sample_count = 0;

}  // namespace



// The name of this function is important for Arduino compatibility.
void setup() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(MobileNetV2_Student_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Reservar el tensor arena en PSRAM (ver comentario junto a su
  // declaración). Si esto devuelve nullptr, revisa que CONFIG_SPIRAM y
  // CONFIG_SPIRAM_MODE_OCT estén habilitados en sdkconfig para esta placa.
  tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM));
  if (tensor_arena == nullptr) {
    MicroPrintf("No se pudo reservar %d bytes en PSRAM para el tensor arena.",
                static_cast<int>(kTensorArenaSize));
    return;
  }


  // Free RAM before interpreter build
  MicroPrintf("[RAM] Free Internal Heap: %u bytes | Free Heap PSRAM: %u bytes",
              (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
              (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));


  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<7> resolver;
  if (resolver.AddConv2D() != kTfLiteOk) return;
  if (resolver.AddDepthwiseConv2D() != kTfLiteOk) return;
  if (resolver.AddAveragePool2D() != kTfLiteOk) return;
  if (resolver.AddMaxPool2D() != kTfLiteOk) return;
  if (resolver.AddReshape() != kTfLiteOk) return;
  if (resolver.AddAdd() != kTfLiteOk) return;
  if (resolver.AddPad() != kTfLiteOk) return;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers of the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  MicroPrintf("Arena realmente usado: %d de %d bytes reservados",
              static_cast<int>(interpreter->arena_used_bytes()),
              static_cast<int>(kTensorArenaSize));


  // Free RAM after AllocateTensors
  MicroPrintf("[RAM] Free Internal Heap: %u bytes | Free Heap PSRAM: %u bytes",
              (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
              (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // Keep track of how many inferences we have performed.
  inference_count = 0;

  // Solo llegamos aquí si TODOS los pasos anteriores tuvieron éxito.
  model_ready = true;
}

// The name of this function is important for Arduino compatibility.
void loop() {

  if (!model_ready) {
    // setup() no terminó con éxito: interpreter/input/output siguen en
    // nullptr. Salir aquí evita el LoadProhibited por desreferenciar un
    // puntero nulo.
    MicroPrintf("Modelo no inicializado, loop() abortado.");
    return;
  }
  
  size_t expected_bytes = input->bytes;

  // Security validation
  if(random_test_sample_quantized_bin_len != expected_bytes){
    MicroPrintf("Error: model's expected bytes: %d, test data has: %d.",
                expected_bytes, random_test_sample_quantized_bin_len);
      return;
  }

  // Measure init
  int64_t t_copy_start_us = esp_timer_get_time();

  // Copy input data to model's input
  memcpy(input->data.int8, random_test_sample_quantized_bin, expected_bytes);

  // Time mark before Invoke()
  int64_t t_invoke_start_us = esp_timer_get_time();

  MicroPrintf("Starting Inference...");

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();

  // Time mark after Invoke()
  int64_t t_invoke_end_us = esp_timer_get_time();

  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }


  // Tensor shape verification
  int num_dims = output->dims->size;
  if (num_dims != 4) {
      MicroPrintf("Critic Error: Expected 4D tensor, tensor dims %dD", num_dims);
      return;
  }

  // Dims extraction
  int batch_size = output->dims->data[0]; // Debería ser 1
  int height     = output->dims->data[1]; // Debería ser 7
  int width      = output->dims->data[2]; // Debería ser 7
  int channels   = output->dims->data[3]; // Debería ser 1280

  MicroPrintf("Succesfull extraction. Feature Map Dimensions: [%d, %d, %d, %d]", 
              batch_size, height, width, channels);

  // Memory and pointer computing
  int total_elements = batch_size * height * width * channels; // 62,720 elementos
  int8_t* feature_maps = output->data.int8; // Puntero al inicio del bloque de memoria

  float scale = output->params.scale;
  int32_t zero_point = output->params.zero_point;

  int target_b = 0;
  int target_h = 0;
  int target_w = 0;

  for (int c = 0; c < 5; ++c) {
      // Fórmula de aplanamiento para indexar un arreglo 1D como si fuera 4D
      int index = ((target_b * height + target_h) * width + target_w) * channels + c;
      
      int8_t quantized_val = feature_maps[index];
      float latent_activation = (quantized_val - zero_point) * scale;
      
      MicroPrintf("  Channel [%d]: INT8 = %d -> Float = %f", 
                  c, quantized_val, static_cast<double>(latent_activation));
  }

  // ==================== Latency Report ====================
  int64_t copy_time_us   = t_invoke_start_us - t_copy_start_us;
  int64_t invoke_time_us = t_invoke_end_us   - t_invoke_start_us;
  int64_t total_time_us  = t_invoke_end_us   - t_copy_start_us;
 
  latency_sample_count += 1;
  latency_sum_us += invoke_time_us;
  if (invoke_time_us < latency_min_us) latency_min_us = invoke_time_us;
  if (invoke_time_us > latency_max_us) latency_max_us = invoke_time_us;
  int64_t latency_avg_us = latency_sum_us / latency_sample_count;
 
  MicroPrintf("[LATENCY] input copy: %lld us | invoke: %lld us | total: %lld us",
              (long long)copy_time_us, (long long)invoke_time_us, (long long)total_time_us);
  MicroPrintf("[LATENCY] invoke -> min: %lld us | max: %lld us | average: %lld us (samples: %u)",
              (long long)latency_min_us, (long long)latency_max_us,
              (long long)latency_avg_us, (unsigned)latency_sample_count);
  // ============================================================================================
  
  // ==================== RAM Report ====================
  UBaseType_t stack_high_water_mark = uxTaskGetStackHighWaterMark(NULL);
  MicroPrintf("[RAM] Remaining Free Stack(high water mark): %u words (~%u bytes)",
              (unsigned)stack_high_water_mark,
              (unsigned)(stack_high_water_mark * sizeof(StackType_t)));
  MicroPrintf("[RAM] Free Internal Heap Now: %u bytes | Free Historic Minimum: %u bytes",
              (unsigned)esp_get_free_heap_size(),
              (unsigned)esp_get_minimum_free_heap_size());
  // ============================================================================================


  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
