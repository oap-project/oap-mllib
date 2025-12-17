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

CLANG_FORMAT=$(which clang-format)
CLANG_FORMAT_VERSION=$($CLANG_FORMAT --version)

if [ -z $CLANG_FORMAT ]; then
    echo clang-format not found, please install it.
    exit 1
fi

echo
echo "Using clang-format ($CLANG_FORMAT) version $CLANG_FORMAT_VERSION ..."
echo "NOTICE: clang-format version should be >= 10.0.0"
echo

if [ ! -f .clang-format ]; then
    echo .clang-format is not found in current directory, please generate it.
    exit 1
fi

ERRORS=$($CLANG_FORMAT --style=file -Werror -n *.cpp *.h 2> /dev/stdout 1> /dev/null | grep error)

if test ! -z "$ERRORS"; then
    echo -e "clang-format style checks failed at following occurrences:\n$ERRORS"
    exit 1
else
    echo -e "clang-format style checks passed."
fi