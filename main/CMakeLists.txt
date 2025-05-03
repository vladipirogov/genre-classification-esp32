idf_component_register(SRCS "main.cc" "MFCC_Q15.cc" "u8g2_esp32_hal.c"
                    INCLUDE_DIRS ".")
#spiffs_create_partition_image(storage ../spiffs FLASH_IN_PROJECT)