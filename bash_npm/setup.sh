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
cd ../;