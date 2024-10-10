#!/bin/bash

# Sync the specified folder to the specified server
# Usage: sync.sh <server> <user>

if [ $# -ne 2 ]; then
    echo "Usage: sync_to_meta.sh <server> <user>"
    echo "Hint: skirit.ics.muni.cz"
    exit 1
fi

ping -c 1 $1 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Metacentrum server $1 is not reachable"
    exit 1
fi

rsync -av --no-perms --exclude data --exclude generated --exclude env . $2@$1:~/synced

if [ $? -ne 0 ]; then
    echo "Syncing this folder to server $1 failed"
    exit 1
fi

echo "This folder successfully synced to server $1"
