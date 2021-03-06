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

if [ -z $CLANG_FORMAT ]; then
    echo clang-format not found, please install it.
    exit 1
fi

$CLANG_FORMAT -style="{ BasedOnStyle: LLVM, \
    UseTab: Never, \
    IndentWidth: 4, \
    TabWidth: 4 \
    }" \
    -dump-config > .clang-format
