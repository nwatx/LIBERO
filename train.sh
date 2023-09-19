#!/usr/bin/bash
pip install -r requirements.txt
python libero/lifelong/main.py benchmark_name=LIBERO_OBJECT policy=bc_rnn_policy lifelong=base