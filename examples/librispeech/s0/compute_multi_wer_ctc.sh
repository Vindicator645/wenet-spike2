decode_dir=$1
mkdir -p $decode_dir/test_clean_context_ctc_greedy_search
mkdir -p $decode_dir/test_other_context_ctc_greedy_search
mkdir -p $decode_dir/test_clean_nocontext_ctc_greedy_search
mkdir -p $decode_dir/test_other_nocontext_ctc_greedy_search
./tools/filter_scp.pl --exclude -f 1 data/test_other_context/context_key_list $decode_dir/test_other_ctc_greedy_search/text > $decode_dir/test_other_nocontext_ctc_greedy_search/text
./tools/filter_scp.pl  -f 1 data/test_other_context/context_key_list $decode_dir/test_other_ctc_greedy_search/text > $decode_dir/test_other_context_ctc_greedy_search/text
python tools/compute-wer.py --char=1 --v=1  data/test_other_context/text $decode_dir/test_other_context_ctc_greedy_search/text > $decode_dir/test_other_context_ctc_greedy_search/wer
python tools/compute-wer.py --char=1 --v=1  data/test_other_nocontext/text $decode_dir/test_other_nocontext_ctc_greedy_search/text > $decode_dir/test_other_nocontext_ctc_greedy_search/wer


./tools/filter_scp.pl --exclude -f 1 data/test_clean_context/context_key_list $decode_dir/test_clean_ctc_greedy_search/text > $decode_dir/test_clean_nocontext_ctc_greedy_search/text
./tools/filter_scp.pl  -f 1 data/test_clean_context/context_key_list $decode_dir/test_clean_ctc_greedy_search/text > $decode_dir/test_clean_context_ctc_greedy_search/text
python tools/compute-wer.py --char=1 --v=1  data/test_clean_context/text $decode_dir/test_clean_context_ctc_greedy_search/text > $decode_dir/test_clean_context_ctc_greedy_search/wer
python tools/compute-wer.py --char=1 --v=1  data/test_clean_nocontext/text $decode_dir/test_clean_nocontext_ctc_greedy_search/text > $decode_dir/test_clean_nocontext_ctc_greedy_search/wer
tail -n 3 $decode_dir/test*_ctc_greedy_search/wer