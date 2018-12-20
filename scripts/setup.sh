#!/usr/bin/env bash

mkdir data;
cd data;

echo "Cleaning datasets..";
rm *.csv *.zip *.txt;

echo "Downloading datasets..";
wget http://uploads.benjamin-raymond.pro/2018/11/23/01-24-01-002-twitter-datasets.zip -O twitter-datasets.zip;
unzip twitter-datasets.zip;
mv twitter-datasets/* ./;
rm -r twitter-datasets;
rm twitter-datasets.zip;

echo "Downloading gloves";
wget http://uploads.benjamin-raymond.pro/2018/12/20/08-00-00-000-glove.twitter-27b-100d.txt -O glove.twitter.27B.100d.txt;

cd ../;
