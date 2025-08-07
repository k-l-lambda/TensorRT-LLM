for file in cubin/*.cu.cubin; do
	out="cubin/$(basename "$file" .cu.cubin).cubin.cpp"
	xxd -i "$file" \
		| sed '1i namespace tensorrt_llm\n{\nnamespace kernels\n{' \
		| sed '$a }\n}' \
		> "$out"
done