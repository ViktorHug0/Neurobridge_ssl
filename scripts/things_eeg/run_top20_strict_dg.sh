#!/bin/bash
set -e

SEED=${SEED:-2025}

CONFIGS=(
"01|0|0.0|"
"02|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256"
"03|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1"
"04|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --style_lambda 0.1"
"05|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.1"
"06|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --decor_lambda 0.01"
"07|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --coral_lambda 0.01"
"08|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.1 --decor_lambda 0.01 --coral_lambda 0.01"
"09|1|0.1|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.1 --decor_lambda 0.01 --coral_lambda 0.01"
"10|1|0.2|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.1 --decor_lambda 0.01 --coral_lambda 0.01"
"11|1|0.5|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.1 --decor_lambda 0.01 --coral_lambda 0.01"
"12|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.05 --style_lambda 0.1 --decor_lambda 0.01 --coral_lambda 0.01"
"13|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.2 --style_lambda 0.1 --decor_lambda 0.01 --coral_lambda 0.01"
"14|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.05 --decor_lambda 0.01 --coral_lambda 0.01"
"15|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.2 --decor_lambda 0.01 --coral_lambda 0.01"
"16|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.1 --decor_lambda 0.005 --coral_lambda 0.01"
"17|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.1 --decor_lambda 0.02 --coral_lambda 0.01"
"18|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.1 --decor_lambda 0.01 --coral_lambda 0.005"
"19|0|0.0|--multi_positive_loss --grouped_batch_sampler --samples_per_image 6 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.1 --decor_lambda 0.01 --coral_lambda 0.02"
"20|1|0.2|--multi_positive_loss --grouped_batch_sampler --samples_per_image 4 --ssl_lambda 1.0 --ssl_projector_dim 256 --adv_lambda 0.1 --style_lambda 0.1 --decor_lambda 0.01 --coral_lambda 0.01"
)

for CFG in "${CONFIGS[@]}"; do
  IFS='|' read -r EXP_ID USE_EPI EPI_LAMBDA EXTRA_ARGS <<< "$CFG"
  echo "Running strict-DG experiment ${EXP_ID}..."
  SEED="$SEED" EPISODIC_SOURCE_HOLDOUT="$USE_EPI" EPISODIC_LAMBDA="$EPI_LAMBDA" EXTRA_ARGS="$EXTRA_ARGS" bash ./scripts/things_eeg/inter-subjects.sh
done

echo "Top-20 strict-DG run set complete."
