
#
# Main component of TF Micro project 'micro_speech'.
#

idf_component_register(
    SRCS main.cc main_functions.cc constants.cc output_handler.cc model.cc test_data.cc
    PRIV_REQUIRES spi_flash esp_timer freertos
    INCLUDE_DIRS "")
