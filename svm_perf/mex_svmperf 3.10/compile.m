
if (ispc)
   % classify
   mex -v GCC='/usr/bin/gcc-4.7'-f mexopts.bat  -DCOMPILE_MEX_INTERFACE -DCOMPILE_STRSEP -output mex_svm_perf_classify  mex_interface.c svm_common.c svm_hideo.c svm_learn.c svm_struct_api.c svm_struct_classify.c svm_struct_common.c svm_struct_learn.c svm_struct_learn_custom.c
   % learn
   mex -v GCC='/usr/bin/gcc-4.7' -f mexopts.bat -DCOMPILE_MEX_INTERFACE -DCOMPILE_STRSEP -output mex_svm_perf_learn mex_interface.c svm_common.c svm_hideo.c svm_learn.c svm_struct_api.c svm_struct_common.c svm_struct_learn.c svm_struct_learn_custom.c svm_struct_main.c
else
   % classify
   fprintf('Compiling svmperfclassify...\n');
   mex -v GCC='/usr/bin/gcc-4.7' CFLAGS="\$CFLAGS -std=c99" -DCOMPILE_MEX_INTERFACE -output mex_svm_perf_classify -largeArrayDims mex_interface.c svm_common.c svm_hideo.c svm_learn.c svm_struct_api.c svm_struct_classify.c svm_struct_common.c svm_struct_learn.c svm_struct_learn_custom.c
   % learn
   fprintf('Compiling svmperflearn...\n');
   mex -v GCC='/usr/bin/gcc-4.7' CFLAGS="\$CFLAGS -std=c99" -DCOMPILE_MEX_INTERFACE -output mex_svm_perf_learn -largeArrayDims mex_interface.c svm_common.c svm_hideo.c svm_learn.c svm_struct_api.c svm_struct_common.c svm_struct_learn.c svm_struct_learn_custom.c svm_struct_main.c
end