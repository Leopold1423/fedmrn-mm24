#!/bin/bash

PROCESS=`ps -ef | grep python | grep main.py | awk '{print $2}' | xargs kill -9`
