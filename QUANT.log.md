```text
Device: cuda

===== FP32 BASELINE =====
FP32 mean L1: 0.10518752

===== INT8 PTQ =====
Inserted 111 quantizers
Calibration samples: 256
stem.input_quantizer                                                             TensorQuantizer(8 bit fake per-tensor amax=2.23e+00 calibrator=MaxCalibrator quant)
stem.output_quantizer                                                            TensorQuantizer(disabled)
stem.weight_quantizer                                                            TensorQuantizer(8 bit fake axis=0 amax=[7.78e-02, 2.13e-01](32) calibrator=MaxCalibrator quant)
contracting_blocks.0.conv1.input_quantizer                                       TensorQuantizer(8 bit fake per-tensor amax=1.11e+01 calibrator=MaxCalibrator quant)
contracting_blocks.0.conv1.output_quantizer                                      TensorQuantizer(disabled)
contracting_blocks.0.conv1.weight_quantizer                                      TensorQuantizer(8 bit fake axis=0 amax=[1.79e-01, 4.79e-01](64) calibrator=MaxCalibrator quant)
...
expanding_blocks.3.skip.input_quantizer                                          TensorQuantizer(8 bit fake per-tensor amax=1.34e+01 calibrator=MaxCalibrator quant)
expanding_blocks.3.skip.output_quantizer                                         TensorQuantizer(disabled)
expanding_blocks.3.skip.weight_quantizer                                         TensorQuantizer(8 bit fake axis=0 amax=[2.72e-02, 2.54e-01](64) calibrator=MaxCalibrator quant)
head.input_quantizer                                                             TensorQuantizer(8 bit fake per-tensor amax=2.32e+00 calibrator=MaxCalibrator quant)
head.output_quantizer                                                            TensorQuantizer(disabled)
head.weight_quantizer                                                            TensorQuantizer(8 bit fake axis=0 amax=[8.16e-02, 9.99e-02](3) calibrator=MaxCalibrator quant)
111 TensorQuantizers found in model
Loading extension modelopt_cuda_ext...
Loaded extension modelopt_cuda_ext in 80.6 seconds
INT8 PTQ mean L1: 0.11849772
PTQ L1 change: +0.01331020 (+12.65%)
Saved PTQ model: unet_int8_ptq.pth

===== INT8 QAT =====
QAT epoch 1/5 - train L1: 0.11686235
QAT epoch 1 mean L1: 0.11497784
QAT epoch 2/5 - train L1: 0.11498316
QAT epoch 2 mean L1: 0.11313606
QAT epoch 3/5 - train L1: 0.11464546
QAT epoch 3 mean L1: 0.11262150
QAT epoch 4/5 - train L1: 0.11384355
QAT epoch 4 mean L1: 0.11233041
QAT epoch 5/5 - train L1: 0.11340119
QAT epoch 5 mean L1: 0.11116522
INT8 QAT mean L1: 0.11116522
QAT L1 change vs FP32: +0.00597770 (+5.68%)
Saved QAT model: unet_int8_qat.pth

===== SUMMARY =====
FP32:    0.10518752
INT8 PTQ: 0.11849772
INT8 QAT: 0.11116522
```
