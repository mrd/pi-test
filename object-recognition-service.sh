#!/bin/bash

DIR=`dirname $0`
LOGFILE=object-recognition.log

cd $DIR

sudo -u pi ./runlcddemo2.sh > $LOGFILE 2>&1

