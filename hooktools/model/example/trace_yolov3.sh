#!/bin/bash

git clone ssh://git@ext-gitlab.denglin.com:23/software/dl_framework/model.git
cd model
bash yolov3-float32-test-tools.sh