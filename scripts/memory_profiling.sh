uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --only_forward \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_128_forward_FP32.pkl \
    --context_length 128 

wait
sleep 5

uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --only_forward \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_256_forward_FP32.pkl \
    --context_length 256

wait
sleep 5

uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --only_forward \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_512_forward_FP32.pkl \
    --context_length 512

wait
sleep 5

uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_128_full_FP32.pkl \
    --context_length 128

wait
sleep 5

uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_256_full_FP32.pkl \
    --context_length 256

wait
sleep 5

uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_512_full_FP32.pkl \
    --context_length 512

wait
sleep 5

uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --only_forward \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_128_forward.pkl \
    --context_length 128 \
    --use_autocast
wait
sleep 5

uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --only_forward \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_256_forward.pkl \
    --context_length 256 \
    --use_autocast

wait
sleep 5

uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --only_forward \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_512_forward.pkl \
    --context_length 512 \
    --use_autocast

wait
sleep 5

uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_128_full.pkl \
    --context_length 128 \
    --use_autocast

wait
sleep 5

uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_256_full.pkl \
    --context_length 256 \
    --use_autocast

wait
sleep 5

uv run python ./scripts/benchmarking.py --model_sizes 2.7B \
    --do_memory_profiling \
    --memory_profiling_file ./results/memory_profile_T_512_full.pkl \
    --context_length 512 \
    --use_autocast

# nohup bash ./scripts/memory_profiling.sh > ./logs/memory_profiling.log 2>&1 &