decode_dir=$1
mkdir -p $decode_dir/test_clean_context_attention_rescoring
mkdir -p $decode_dir/test_other_context_attention_rescoring
mkdir -p $decode_dir/test_clean_nocontext_attention_rescoring
mkdir -p $decode_dir/test_other_nocontext_attention_rescoring
./tools/filter_scp.pl --exclude -f 1 data/test_other_context/context_key_list $decode_dir/test_other_attention_rescoring/text > $decode_dir/test_other_nocontext_attention_rescoring/text
./tools/filter_scp.pl  -f 1 data/test_other_context/context_key_list $decode_dir/test_other_attention_rescoring/text > $decode_dir/test_other_context_attention_rescoring/text
python tools/compute-wer.py --char=1 --v=1  data/test_other_context/text $decode_dir/test_other_context_attention_rescoring/text > $decode_dir/test_other_context_attention_rescoring/wer
python tools/compute-wer.py --char=1 --v=1  data/test_other_nocontext/text $decode_dir/test_other_nocontext_attention_rescoring/text > $decode_dir/test_other_nocontext_attention_rescoring/wer


./tools/filter_scp.pl --exclude -f 1 data/test_clean_context/context_key_list $decode_dir/test_clean_attention_rescoring/text > $decode_dir/test_clean_nocontext_attention_rescoring/text
./tools/filter_scp.pl  -f 1 data/test_clean_context/context_key_list $decode_dir/test_clean_attention_rescoring/text > $decode_dir/test_clean_context_attention_rescoring/text
python tools/compute-wer.py --char=1 --v=1  data/test_clean_context/text $decode_dir/test_clean_context_attention_rescoring/text > $decode_dir/test_clean_context_attention_rescoring/wer
python tools/compute-wer.py --char=1 --v=1  data/test_clean_nocontext/text $decode_dir/test_clean_nocontext_attention_rescoring/text > $decode_dir/test_clean_nocontext_attention_rescoring/wer
tail -n 3 $decode_dir/test*_attention_rescoring/wer