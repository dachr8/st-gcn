#!/bin/bash

bash /content/st-gcn/tools/get_models.sh

pip install -r /content/st-gcn/requirements.txt
pip install --upgrade pyyaml

cd /content/st-gcn/torchlight || exit
python setup.py install