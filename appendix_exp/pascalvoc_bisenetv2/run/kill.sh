#!/bin/bash

PROCESS=`ps -ef | grep python | grep main.py | grep "$1" | awk '{print $2}' | xargs kill -9`

