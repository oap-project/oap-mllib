#!/usr/bin/env bash

# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [[ $OAP_MLLIB_TESTING == "true" ]]; then
  exit 0
fi

if [[ -z $ONEDAL_VERSION ]]; then
  if [[ -e $DAALROOT ]]; then
    export ONEDAL_VERSION=$(echo "$DAALROOT" | awk -F '/' '{print $(NF)}')
  elif [[ -e $DALROOT ]]; then
    export ONEDAL_VERSION=$(echo "$DALROOT" | awk -F '/' '{print $(NF)}')
  else
    echo DAALROOT not defined!
    exit 1
  fi
  echo $ONEDAL_VERSION
fi

# Function to compare version strings
compare_versions() {
    local v1=$1
    local v2=$2

    # Convert versions to arrays
    IFS='.' read -ra v1_array <<< "$v1"
    IFS='.' read -ra v2_array <<< "$v2"

    # Iterate through each segment and compare numerically
    for i in {0..2}; do
        if ((v1_array[i] > v2_array[i])); then
            return 0  # v1 > v2
        elif ((v1_array[i] < v2_array[i])); then
            return 1  # v1 < v2
        fi
    done

    return 1  # v1 == v2
}

# Reference version
reference_version="2023.2.0"

# Compare versions
compare_versions "$ONEDAL_VERSION" "$reference_version"
result=$?

# Check the result of the comparison
if [ "$result" -eq 0 ]; then
    echo "$ONEDAL_VERSION is greater than $reference_version"
    make clean
    make  -f Makefile -j
elif [ "$result" -eq 1 ]; then
    echo "$ONEDAL_VERSION is less than or equal $reference_version"
    make clean
    make  -f Makefile_2023.2.0 -j

fi
