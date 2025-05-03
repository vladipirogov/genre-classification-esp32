#include "esp_log.h"
#include "driver/adc.h"
#include "esp_adc_cal.h"
#include "esp_timer.h"
#include <esp_heap_caps.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "model.h"

#include "MFCC_Q15.h"

#include "u8g2_esp32_hal.h"

// Constants
#define DEFAULT_VREF        1100
#define SAMPLE_RATE         22050
#define AUDIO_LENGTH_SEC    1
#define AUDIO_LENGTH_SAMPLES (SAMPLE_RATE * AUDIO_LENGTH_SEC)
#define TAG                 "GENRE_CLASSIFICATION"
#define PIN_SDA             GPIO_NUM_5
#define PIN_SCL             GPIO_NUM_6

//Display variable
u8g2_t u8g2;

// Labels
static const char *label[] = {"blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"};

// TensorFlow Lite variables
const tflite::Model* tflu_model = nullptr;
tflite::MicroInterpreter* tflu_interpreter = nullptr;
TfLiteTensor* tflu_i_tensor = nullptr;
TfLiteTensor* tflu_o_tensor = nullptr;
constexpr int tensor_arena_size = 70 * 1024;
uint8_t *tensor_arena;

// Audio buffer
volatile int16_t audio_buffer[AUDIO_LENGTH_SAMPLES];
volatile int32_t buffer_index = 0;
volatile bool buffer_ready = false;
static esp_adc_cal_characteristics_t* adc_chars;
int bias_offset = 1552;

static MFCC_Q15 mfccs;


/**
 * Timer callback for audio sampling
 * This function is called periodically to read audio samples from the ADC.
 */
void timer_callback(void* arg) {
    if (buffer_index < AUDIO_LENGTH_SAMPLES) {
        int adc_value = adc1_get_raw(ADC1_CHANNEL_0); // Read ADC
        //printf("Raw ADC Value: %d\n", adc_value);
        audio_buffer[buffer_index++] = adc_value - bias_offset; // Adjust for bias
    } else {
        buffer_ready = true;
    }
}

/**
 * ADC calibration function to measure bias offset
 * @return Average ADC value over a number of samples
 */
int measure_bias_offset()
{
    int sum = 0;
    const int samples = 1000;
    for (int i = 0; i < samples; i++) {
        sum += adc1_get_raw(ADC1_CHANNEL_0);
        ets_delay_us(1000); 
    }
    return sum / samples;
}

/**
 * TensorFlow Lite initialization
 */
void tflu_initialization() {
    if (tensor_arena == NULL) {
        tensor_arena = (uint8_t *) heap_caps_malloc(tensor_arena_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
      }
      if (tensor_arena == NULL) {
        ESP_LOGE(TAG, "Couldn't allocate memory of %d bytes\n", tensor_arena_size);
        return;
      } 

    tflu_model = tflite::GetModel(model_tflite);
    if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch!");
        while (1);
    }

    static tflite::MicroMutableOpResolver<9> resolver;
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddSub();
    resolver.AddMul();
    resolver.AddUnidirectionalSequenceLSTM();
    resolver.AddStridedSlice();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    static tflite::MicroInterpreter static_interpreter(
        tflu_model, resolver, tensor_arena, tensor_arena_size);

    tflu_interpreter = &static_interpreter;
    tflu_interpreter->AllocateTensors();

    tflu_i_tensor = tflu_interpreter->input(0);
    tflu_o_tensor = tflu_interpreter->output(0);

    ESP_LOGI(TAG, "TensorFlow Lite initialization completed");
}

/**
 * Audio processing task
 * This task runs in a loop, waiting for the audio buffer to be filled,
 * computing MFCCs, and running inference on the TensorFlow Lite model.
 * It also updates the display with the predicted genre.
 */
void audio_processing_task(void* arg) {
    while (1) {
        // Wait for buffer to be ready
        if (!buffer_ready) {
            vTaskDelay(pdMS_TO_TICKS(10)); // Yield to other tasks
            continue;
        }

        // Compute MFCCs
        mfccs.run((const q15_t*)audio_buffer, tflu_i_tensor->data.f);

        // Run inference
        if (tflu_interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Error invoking TensorFlow Lite interpreter");
            buffer_index = 0;
            buffer_ready = false;
            continue;
        }

        // Get the predicted genre
        size_t max_index = 0;
        float max_value = 0;
        for (size_t i = 0; i < 10; i++) {
            if (tflu_o_tensor->data.f[i] > max_value) {
                max_index = i;
                max_value = tflu_o_tensor->data.f[i];
            }
        }

        ESP_LOGI(TAG, "Predicted genre: %s", label[max_index]);

        u8g2_ClearBuffer(&u8g2);
		u8g2_SetFont(&u8g2, u8g2_font_ncenB08_tr);
		u8g2_DrawStr(&u8g2, 0,10, label[max_index]);
		u8g2_SendBuffer(&u8g2);

        // Reset buffer for the next cycle
        buffer_index = 0;
        buffer_ready = false;
    }
}

static void print_char_val_type(esp_adc_cal_value_t val_type)
{
    if (val_type == ESP_ADC_CAL_VAL_EFUSE_TP) {
        printf("Characterized using Two Point Value\n");
    } else if (val_type == ESP_ADC_CAL_VAL_EFUSE_VREF) {
        printf("Characterized using eFuse Vref\n");
    } else {
        printf("Characterized using Default Vref\n");
    }
}

// Main application
extern "C" void app_main() {
    // initialize the u8g2 hal
	u8g2_esp32_hal_t u8g2_esp32_hal = U8G2_ESP32_HAL_DEFAULT;
	u8g2_esp32_hal.sda = PIN_SDA;
	u8g2_esp32_hal.scl = PIN_SCL;
	u8g2_esp32_hal_init(u8g2_esp32_hal);

	// initialize the u8g2 library
	u8g2_Setup_ssd1306_i2c_72x40_er_f(
		&u8g2,
		U8G2_R0,
		u8g2_esp32_i2c_byte_cb,
		u8g2_esp32_gpio_and_delay_cb);
	
	// set the display address
	u8x8_SetI2CAddress(&u8g2.u8x8, 0x78);
	
	// initialize the display
	u8g2_InitDisplay(&u8g2);
	
	// wake up the display
	u8g2_SetPowerSave(&u8g2, 0);

    // Initialize ADC
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_0, ADC_ATTEN_DB_11);
    adc_chars = (esp_adc_cal_characteristics_t*) calloc(1, sizeof(esp_adc_cal_characteristics_t));
    if (adc_chars == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for ADC characteristics");
        return;
    }
    esp_adc_cal_value_t val_type = esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, DEFAULT_VREF, adc_chars);
    print_char_val_type(val_type);
    printf("Measuring bias offset...\n");
    bias_offset = measure_bias_offset();
    printf("Bias offset measured: %d\n", bias_offset);

    // Initialize TensorFlow Lite
    tflu_initialization();

    // Create a periodic timer for audio sampling
    const esp_timer_create_args_t timer_args = {
        .callback = &timer_callback,
        .name = "audio_timer"
    };
    esp_timer_handle_t timer;
    esp_timer_create(&timer_args, &timer);
    esp_timer_start_periodic(timer, 1000000 / SAMPLE_RATE); // Sampling rate

    // Create the audio processing task
    xTaskCreate(audio_processing_task, "AudioProcessingTask", 4096, NULL, 5, NULL);

    // app_main should not block; let FreeRTOS handle tasks
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000)); // Keep the main task alive
    }
}