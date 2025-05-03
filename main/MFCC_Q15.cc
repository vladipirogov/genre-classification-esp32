#include "MFCC_Q15.h"

MFCC_Q15::MFCC_Q15() {
    // Initialize RFFT instance
    arm_rfft_init_q15(&_rfft_inst, FFT_LENGTH, 0, 1);

    // Initialize Mel-weight matrix instance
    _mel_wei_mtx_inst.numRows = mel_wei_mtx_q15_T_dim1;
    _mel_wei_mtx_inst.numCols = mel_wei_mtx_q15_T_dim0;
    _mel_wei_mtx_inst.pData = (q15_t *)&mel_wei_mtx_q15_T_data[0];

    // Initialize DCT-weight matrix instance
    _dct_wei_mtx_inst.numRows = dct_wei_mtx_q15_T_dim1;
    _dct_wei_mtx_inst.numCols = dct_wei_mtx_q15_T_dim0;
    _dct_wei_mtx_inst.pData = (q15_t *)&dct_wei_mtx_q15_T_data[0];
}

void MFCC_Q15::run(const q15_t* src, float* dst) {
    for (int i = 0; i < NUM_FRAMES; ++i) {
        // Apply the Hann window
        arm_mult_q15((q15_t*)&src[i * FRAME_STEP], (q15_t*)hann_lut_q15_data, _bufA, FRAME_LENGTH);

        // Calculate the RFFT
        arm_rfft_q15(&_rfft_inst, _bufA, _bufB);

        // Calculate the magnitude
        _bufA[0]                 = _bufB[0];
        _bufA[NUM_FFT_FREQS - 1] = _bufB[1];
        arm_cmplx_mag_q15(&_bufB[2], &_bufA[1], NUM_FFT_FREQS - 2);

        // Mel-scale conversion
        arm_mat_vec_mult_q15(&_mel_wei_mtx_inst, _bufA, _bufB);

        for (int idx = 0; idx < NUM_MEL_FREQS; ++idx) {
            const int16_t val = (int16_t)_bufB[idx];
            _bufA[idx] = log_lut_q13_3_data[val];
        }

        // Calculate the MFCCs through the DCT
        arm_mat_vec_mult_q15(&_dct_wei_mtx_inst, _bufA, _bufB);

        for (int k = 0; k < NUM_MFCCS; ++k) {
            dst[k + i * NUM_MFCCS] = (float)_bufB[k] / (float)(8);
        }
    }
}