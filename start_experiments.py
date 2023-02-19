from elliot.run import run_experiment

config = """experiment:
  backend: pytorch
  path_output_rec_result: ./results/{0}/topk/recs/
  path_output_rec_weight: ./results/{0}/topk/weights/
  path_output_rec_performance: ./results/{0}/topk/performance/
  data_config:
    strategy: fixed
    train_path: ../data/{0}/Train.tsv
    validation_path: ../data/{0}/Val.tsv
    test_path: ../data/{0}/Test.tsv
    side_information:
      - dataloader: SentimentInteractionsTextualAttributesUUII
        uu_dot: ../data/{0}/uu_dot_topk.npz
        uu_max: ../data/{0}/uu_max_topk.npz
        uu_min: ../data/{0}/uu_min_topk.npz
        uu_avg: ../data/{0}/uu_avg_topk.npz
        ii_dot: ../data/{0}/ii_dot_topk.npz
        ii_max: ../data/{0}/ii_max_topk.npz
        ii_min: ../data/{0}/ii_min_topk.npz
        ii_avg: ../data/{0}/ii_avg_topk.npz
        uu_rat_dot: ../data/{0}/uu_rat_dot_topk.npz
        uu_rat_max: ../data/{0}/uu_rat_max_topk.npz
        uu_rat_min: ../data/{0}/uu_rat_min_topk.npz
        uu_rat_avg: ../data/{0}/uu_rat_avg_topk.npz
        uu_no_coeff: ../data/{0}/uu_no_coeff_topk.npz
        uu_rat_no_coeff: ../data/{0}/uu_rat_no_coeff_topk.npz
        ii_rat_dot: ../data/{0}/ii_rat_dot_topk.npz
        ii_rat_max: ../data/{0}/ii_rat_max_topk.npz
        ii_rat_min: ../data/{0}/ii_rat_min_topk.npz
        ii_rat_avg: ../data/{0}/ii_rat_avg_topk.npz
        ii_no_coeff: ../data/{0}/ii_no_coeff_topk.npz
        ii_rat_no_coeff: ../data/{0}/ii_rat_no_coeff_topk.npz
  dataset: Digital_Music
  top_k: 20
  evaluation:
    cutoffs: [20]
    simple_metrics: [Recall, MSE]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.UUIIGCMC:
      meta:
        hyper_opt_alg: grid
        verbose: True
        save_weights: False
        save_recs: False
        validation_rate: 1
        validation_metric: MSE
        restore: False
      epochs: 50
      batch_size: 128
      factors: 32
      lr: [0.0001, 0.0005, 0.001, 0.005, 0.01]
      n_uu: [1, 2, 3, 4]
      a: [0.0, 0.3, 0.7, 1.0]
      sim_uu: [dot, max, min, avg, rat_dot, rat_max, rat_min, rat_avg, no_coeff, rat_no_coeff]
      conv_size: 32
      dense_size: 32
      n_conv: 1
      n_dense: 1
      relations: (1,2,3,4,5)
      num_basis: 2
      dropout: 0.1
      acc: stack
      loader: ('SentimentInteractionsTextualAttributesUUII',)
      seed: 123
"""

topk = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for t in topk:
    with open(f'config_files/test_error_uuii_{t}.yml', 'w') as f:
        f.write(config.replace('topk', str(t)))
    print(f'**********EXPERIMENT WITH TOP-K: {t} HAS STARTED!**********')
    run_experiment(f"config_files/test_error_uuii_{t}.yml")
    print(f'**********EXPERIMENT WITH TOP-K: {t} HAS ENDED!**********')
