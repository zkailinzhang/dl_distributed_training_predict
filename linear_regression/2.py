# -*- coding:utf-8 -*-

# Polyaxon
from polyaxon_client.tracking import Experiment

# Polyaxon
experiment = Experiment()

from 1 import x_train

arg

a = prepare(args)

# Polyaxon
experiment.log_data_ref(data=x_train, data_name='dataset_X')
experiment.log_data_ref(data=y_train, data_name='dataset_y')


# learn
resutl = train(a)

# Polyaxon
#这个参数可以自定义吗
experiment.log_metrics(R2_score_train=R2_score_train, R2_score_val=R2_score_verify)

