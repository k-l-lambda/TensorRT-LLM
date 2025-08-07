#!/bin/bash

SRC_DIR=./fmha_v2_cu
OUT_DIR=cubin
mkdir -p "$OUT_DIR"

for src in "$SRC_DIR"/*.cu; do
	# 1. 从文件名里抓 _sm\d\d
	sm_arch=$(basename "$src" | grep -oE '_sm[0-9]+' | head -n1)  # 如 _sm90
	if [ -z "$sm_arch" ]; then
		echo "ERROR: no _smNN found in $src"
		continue
	fi
	sm_num=${sm_arch#_sm}      # 去掉 _sm 前缀，只剩数字，如 90

	# 2. 拼出 compute_XX 和 sm_XX
	compute_arch="compute_${sm_num}"
	code_sm="sm_${sm_num}"

	# 3. 输出路径
	out="$OUT_DIR/$(basename "$src").cubin"

	# 4. 编译
	/usr/local/cuda/bin/nvcc \
		-O3 -std=c++17 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr \
		-gencode=arch="$compute_arch",code="$code_sm" \
		--keep --keep-dir ./temp \
		-g -lineinfo \
		-Xcicc --uumn -Xptxas -uumn \
		-DFMHA_ENABLE_SM89_QMMA \
		-DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE \
		-DHAVE_HALF_ACCUMULATION_FOR_FLASH_ATTENTION \
		-DGENERATE_CUBIN \
		-DUSE_I2F_EMULATION_TRICK \
		-DUSE_F2I_EMULATION_TRICK \
		-I/code/tensorrt_llm/cpp \
		-I/code/tensorrt_llm/cpp/include \
		-I/code/tensorrt_llm/cpp/kernels/fmha_v2/src \
		-I./fmha_v2_cu \
		-I/usr/local/cuda/include \
		-cubin \
		-o "$out" \
		"$src"
done
