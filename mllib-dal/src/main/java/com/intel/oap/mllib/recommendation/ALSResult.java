/*
 * Copyright 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.oap.mllib.recommendation;

public class ALSResult {
  private long rankId = -1;
  private long cUsersFactorsNumTab;
  private long cItemsFactorsNumTab;
  private long cUserOffset;
  private long cItemOffset;

  public long getRankId() {
    return rankId;
  }

  public void setRankId(long rankId) {
    this.rankId = rankId;
  }

  public long getcUsersFactorsNumTab() {
    return cUsersFactorsNumTab;
  }

  public void setcUsersFactorsNumTab(long cUsersFactorsNumTab) {
    this.cUsersFactorsNumTab = cUsersFactorsNumTab;
  }

  public long getcItemsFactorsNumTab() {
    return cItemsFactorsNumTab;
  }

  public void setcItemsFactorsNumTab(long cItemsFactorsNumTab) {
    this.cItemsFactorsNumTab = cItemsFactorsNumTab;
  }

  public long getcUserOffset() {
    return cUserOffset;
  }

  public void setcUserOffset(long cUserOffset) {
    this.cUserOffset = cUserOffset;
  }

  public long getcItemOffset() {
    return cItemOffset;
  }

  public void setcItemOffset(long cItemOffset) {
    this.cItemOffset = cItemOffset;
  }
}
