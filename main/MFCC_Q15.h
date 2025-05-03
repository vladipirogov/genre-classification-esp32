#ifndef MFCC_Q15_H
#define MFCC_Q15_H

#include "arm_math.h"
#include "mfccs_consts.h"
#include "dct_wei_mtx_q15_T.h"
#include "hann_lut_q15.h"
#include "log_lut_q13_3.h"
#include "mel_wei_mtx_q15_T.h"

class MFCC_Q15 {
public:
    MFCC_Q15();
    void run(const q15_t* src, float* dst);

private:
    arm_rfft_instance_q15   _rfft_inst;
    arm_matrix_instance_q15 _mel_wei_mtx_inst;
    arm_matrix_instance_q15 _dct_wei_mtx_inst;
    q15_t                   _bufA[FRAME_LENGTH];
    q15_t                   _bufB[FRAME_LENGTH * 2];
};

#endif // MFCC_Q15_H