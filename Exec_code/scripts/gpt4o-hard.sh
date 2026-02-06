DATA_ROOT=datasets
outdir=${OUTDIR:-${DATA_ROOT}/exprs_map/test/}

flag="--root_dir ${DATA_ROOT}
      --img_root RGB_Observations
      --split NavBench_Hard
      --end 144  # the number of cases to be tested
      --output_dir ${outdir}
      --max_action_len 20
      --stop_after 3
      --llm gpt-4o
      --response_format json
      --max_tokens 1000
      "

python main_gpt.py $flag
