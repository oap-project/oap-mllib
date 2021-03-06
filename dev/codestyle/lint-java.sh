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

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

MVN=$(which mvn)

if [ -z $MVN ]; then
    echo Apache Maven not found, please install it.
    exit 1
fi

ERRORS=$($MVN checkstyle:check | grep ERROR)

if test ! -z "$ERRORS"; then
    echo -e "Checkstyle checks failed at following occurrences:\n$ERRORS"
    exit 1
else
    echo -e "Checkstyle checks passed."
fi
