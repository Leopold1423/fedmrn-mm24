#!/bin/bash

PROCESS=`ps -ef  | grep lishiwei | grep python | grep server.py | awk '{print $2}' | xargs kill -9`
PROCESS=`ps -ef  | grep lishiwei | grep python | grep client.py | awk '{print $2}' | xargs kill -9`
