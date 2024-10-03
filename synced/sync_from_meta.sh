#!/bin/bash

# Sync the generated folder (~/synced/generated) to this folder
# Usage: sync_from_meta.sh <server> <user>

if [ $# -ne 2 ]; then
    echo "Usage: sync_from_meta.sh <server>"
    echo "Hint: skirit.ics.muni.cz"
    exit 1
fi

ping -c 1 $1 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Metacentrum server $1 is not reachable"
    exit 1
fi

rsync -av --no-perms $2@$1:~/synced/generated .

if [ $? -ne 0 ]; then
    echo "Syncing generated folder from server $1 failed"
    exit 1
fi

echo "This folder successfully synced from server $1"
