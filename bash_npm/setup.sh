#!/usr/bin/env bash

echo "Downloading datasets..";
mkdir data;
cd data;
wget http://uploads.benjamin-raymond.pro/2018/11/23/01-24-01-000-sample_submission.csv -O sample_submission.csv;
wget http://uploads.benjamin-raymond.pro/2018/11/23/01-24-01-001-test_data.txt -O test_data.txt;
wget http://uploads.benjamin-raymond.pro/2018/11/23/01-24-01-002-twitter-datasets.zip -O twitter-datasets.zip;
cd ../;