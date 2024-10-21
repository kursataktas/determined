import { V1ColumnType } from 'services/api-ts-sdk';
import { ProjectColumn } from 'types';

export const projectColumns: ProjectColumn[] = [
  {
    column: 'id',
    displayName: 'ID',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'name',
    displayName: 'Name',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'state',
    displayName: 'State',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'startTime',
    displayName: 'Start time',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_DATE',
  },
  {
    column: 'user',
    displayName: 'User',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'numTrials',
    displayName: 'Trial count',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'searcherType',
    displayName: 'Searcher',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'searcherMetric',
    displayName: 'Searcher Metric',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'searcherMetricsVal',
    displayName: 'Searcher Metric Value',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'description',
    displayName: 'Description',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'tags',
    displayName: 'Tags',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'forkedFrom',
    displayName: 'Forked',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'progress',
    displayName: 'Progress',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'duration',
    displayName: 'Duration',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'resourcePool',
    displayName: 'Resource pool',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'checkpointCount',
    displayName: 'Checkpoints',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'checkpointSize',
    displayName: 'Checkpoint size',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'externalExperimentId',
    displayName: 'External Experiment ID',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'externalTrialId',
    displayName: 'External Trial ID',
    location: 'LOCATION_TYPE_EXPERIMENT',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'externalRunId',
    displayName: 'External Run ID',
    location: 'LOCATION_TYPE_RUN',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'experimentName',
    displayName: 'Experiment Name',
    location: 'LOCATION_TYPE_RUN',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'training.☃.last',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'training.☃.max',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'training.☃.mean',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'training.☃.min',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'training.acc.last',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'training.acc.max',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'training.acc.mean',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'training.acc.min',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'training.accuracy.last',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'training.accuracy.max',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'training.accuracy.mean',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'training.accuracy.min',
    displayName: '',
    location: 'LOCATION_TYPE_TRAINING',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_0.last',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_0.max',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_0.mean',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_0.min',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_1.last',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_1.max',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_1.mean',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_1.min',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_2.last',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_2.max',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_2.mean',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'group_b.group_b/metric_2.min',
    displayName: '',
    location: 'LOCATION_TYPE_CUSTOM_METRIC',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.acc.last',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.acc.max',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.acc.mean',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.acc.min',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.accuracy.last',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.accuracy.max',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.accuracy.mean',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.accuracy.min',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.accuracy_baseline.last',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.accuracy_baseline.max',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.accuracy_baseline.mean',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'validation.accuracy_baseline.min',
    displayName: '',
    location: 'LOCATION_TYPE_VALIDATIONS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.global_batch_size',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.layer1_dropout',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.layer2_dropout',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.layer3_dropout',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.learning_rate',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.learning_rate_decay',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.dropout1',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.dropout2',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.n_filters1',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.n_filters2',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.hidden_size',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.l1_regularization',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.l2_regularization',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.max_depth',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.min_node_weight',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.n_trees',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.aux_loss',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.backbone',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.backend',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.bbox_loss_coef',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.cat_ids',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.clip_max_norm',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.cls_loss_coef',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.data_dir',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.dataset_file',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.dec_layers',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.dec_n_points',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.device',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.dice_loss_coef',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.dilation',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.dim_feedforward',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.dropout',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.enc_layers',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.enc_n_points',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.focal_alpha',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.giou_loss_coef',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.hidden_dim',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.lr',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.lr_backbone',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.lr_backbone_names',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.lr_drop',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.lr_linear_proj_mult',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.lr_linear_proj_names',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.mask_loss_coef',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.masks',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.nheads',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.num_classes',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.num_feature_levels',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.num_queries',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.num_workers',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.position_embedding',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.set_cost_bbox',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.set_cost_class',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.set_cost_giou',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.sgd',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.two_stage',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.warmstart',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.weight_decay',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.with_box_refine',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.layer1_dense_size',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.backbone_name',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.classifier.learning_rates',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.classifier.logit_clipping.alpha',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.classifier.logit_clipping.enabled',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.classifier.logit_regularization_beta',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.classifier.momentum',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.classifier.train_epochs',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.self_supervised.lars_eta',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.self_supervised.learning_rate.base',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.self_supervised.learning_rate.base_batch_size',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.self_supervised.learning_rate.warmup_epochs',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.self_supervised.momentum',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.self_supervised.moving_average_decay_base',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.self_supervised.weight_decay',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.training_mode',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.validate_with_classifier',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.deepspeed_config',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.ep_world_size',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.min_capacity',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.moe',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.moe_param_group',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.noisy_gate_policy',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.num_experts',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.top_k',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.data_workers',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.discriminator_width_base',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.generator_width_base',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.noise_length',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.increment_by',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.model.pretrained',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.optimizer.clip_grad',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.optimizer.lr',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.optimizer.momentum',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.optimizer.opt',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.optimizer.opt_betas',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.optimizer.opt_eps',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.optimizer.weight_decay',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.transform.auto_augment',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.transform.color_jitter',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.transform.hflip',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.transform.interpolation',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.transform.ratio',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.transform.re_count',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.transform.re_mode',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.transform.re_prob',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.transform.scale',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.transform.vflip',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.arch',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.data_location',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.dataset',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.evaluate',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.momentum',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.pretrained',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.workers',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.clip_grads',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.config_file',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.gamma',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.step1',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.step2',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.warmup',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.warmup_iters',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.warmup_ratio',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.hidden_layer_1',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.hidden_layer_2',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.hidden_layer_3',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.test1.test2.test3.test4.optimizer_fake.lr',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.test1.test2.test3.test4.optimizer_fake.momentum',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.metrics_base',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.metrics_progression',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.metrics_sigma',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.scheduler',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.dense1',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.adam_epsilon',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.cache_dir',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.lr_scheduler_type',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.model_mode',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.num_warmup_steps',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.pretrained_model_name_or_path',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.use_apex_amp',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.use_pretrained_weights',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.dataset_len',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.model.dropout1',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.trainer.train_metric_agg_rate',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.height_shift_range',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.horizontal_flip',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.width_shift_range',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.gain_per_batch',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.starting_base_value',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.training_structure',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.validation_structure',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.fail',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.irrelevant',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.as_rgb',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.data_flag',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.model_flag',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.num_epochs',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.resize',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.task',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.download',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.doc_stride',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.lr_scheduler_epoch_freq',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.max_answer_length',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.max_grad_norm',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.max_query_length',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.max_seq_length',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.model_type',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.n_best_size',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.null_score_diff_threshold',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.num_training_steps',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.topk_pooling_ratio',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.alpha',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.dropout3',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.n_layers',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.normalize',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.target_label',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.epochs',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.dropout_rate',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.also_irrelevant',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.project',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.workspace',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.some_list',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.height_factor',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.width_factor',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.value',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.beta1',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.beta2',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.conf_dir',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_TEXT',
  },
  {
    column: 'hp.conf_file',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.eval_tasks',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.overwrite_values.model_parallel_size',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.overwrite_values.pipe_parallel_size',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.overwrite_values.train_batch_size',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.overwrite_values.train_micro_batch_size_per_gpu',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.search_world_size',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.user_script',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.wandb_group',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.wandb_team',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.categories',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.categories.animals',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_UNSPECIFIED',
  },
  {
    column: 'hp.categories.legs',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.irrelevant1',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'hp.irrelevant2',
    displayName: '',
    location: 'LOCATION_TYPE_HYPERPARAMETERS',
    type: 'COLUMN_TYPE_NUMBER',
  },
  {
    column: 'metadata.hello',
    displayName: '',
    location: 'LOCATION_TYPE_RUN_METADATA',
    type: 'COLUMN_TYPE_TEXT',
  },
];

export const initialVisibleColumns: [V1ColumnType, string][] = [
  [V1ColumnType.NUMBER, 'id'],
  [V1ColumnType.TEXT, 'name'],
  [V1ColumnType.TEXT, 'state'],
  [V1ColumnType.DATE, 'startTime'],
  [V1ColumnType.TEXT, 'user'],
  [V1ColumnType.NUMBER, 'numTrials'],
  [V1ColumnType.TEXT, 'searcherType'],
  [V1ColumnType.TEXT, 'searcherMetric'],
  [V1ColumnType.NUMBER, 'searcherMetricsVal'],
  [V1ColumnType.TEXT, 'description'],
  [V1ColumnType.TEXT, 'tags'],
  [V1ColumnType.NUMBER, 'progress'],
  [V1ColumnType.NUMBER, 'duration'],
  [V1ColumnType.TEXT, 'resourcePool'],
  [V1ColumnType.NUMBER, 'checkpointCount'],
  [V1ColumnType.NUMBER, 'checkpointSize'],
];
