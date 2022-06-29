# network architecture
# encoder related
encoder: conformer
encoder_conf:
    output_size: 256    # dimension of attention
    attention_heads: 4
    linear_units: 2048  # the number of units of position-wise feed forward
    num_blocks: 12      # the number of encoder blocks
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: conv2d # encoder input type, you can chose conv2d, conv2d6 and conv2d8
    normalize_before: true
    cnn_module_kernel: 15
    use_cnn_module: True
    activation_type: 'swish'
    pos_enc_layer_type: 'rel_pos'
    selfattention_layer_type: 'rel_selfattn'

# decoder related
decoder: transformer
decoder_conf:
    attention_heads: 4
    linear_units: 2048
    num_blocks: 6
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    self_attention_dropout_rate: 0.0
    src_attention_dropout_rate: 0.0

# context related
context: nnbias
context_conf:
    embedding_size: 256
    num_layers: 2
    attention_heads: 4
    linear_units: 512
    num_block: 2
    dropout_rate: 0.0

# hybrid CTC/attention
model_conf:
    ctc_weight: 0.3
    lsm_weight: 0.1     # label smoothing option
    length_normalized_loss: false

# dataset related
dataset_conf:
    filter_conf:
        max_length: 2000
        min_length: 50
        token_max_length: 400
        token_min_length: 1
        min_output_input_ratio: 0.0005
        max_output_input_ratio: 0.1
    resample_conf:
        resample_rate: 16000
    speed_perturb: false
    fbank_conf:
        num_mel_bins: 80
        frame_shift: 10
        frame_length: 25
        dither: 0.0
    spec_aug: true
    spec_aug_conf:
        num_t_mask: 2
        num_f_mask: 2
        max_t: 50
        max_f: 10
    shuffle: true
    shuffle_conf:
        shuffle_size: 1500
    sort: true
    sort_conf:
        sort_size: 500  # sort_size should be less than shuffle_size
    batch_conf:
        batch_type: 'dynamic' # static or dynamic
        batch_size: 6
    context_mode: 1
    bpe_dict: 'data/lang_char/train_960_unigram5000_units.txt'
    pad_conf:
        context_len_min: 1
        context_len_max: 3
        context_list_valid: 'data/processed_context_valid'
        context_list_test: 'data/processed_context_test'  

grad_clip: 5
accum_grad: 1
max_epoch: 120
log_interval: 100

optim: adam
optim_conf:
    lr: 0.004
scheduler: warmuplr     # pytorch v1.1.0+ required
scheduler_conf:
    warmup_steps: 25000
