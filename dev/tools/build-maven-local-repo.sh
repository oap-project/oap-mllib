#!/usr/bin/env bash

if [[ -z $DAALROOT ]]; then
  echo DAALROOT not defined!
  exit 1
fi

echo "Building Maven Repo for oneDAL ..."

mkdir maven-repository
mvn deploy:deploy-file -Dfile=$DAALROOT/lib/onedal.jar -DgroupId=com.intel.onedal -Dversion=2021.4.0 -Dpackaging=jar -Durl=file:./maven-repository -DrepositoryId=maven-repository -DupdateReleaseInfo=true

echo "DONE"

find ./maven-repository

# Add the following into pom.xml:

# <repositories>
#     <repository>
#         <id>maven-repository</id>
#         <url>file:///${project.basedir}/maven-repository</url>
#     </repository>
# </repositories>

# <dependency>
#     <groupId>com.intel.dal</groupId>
#     <artifactId>dal</artifactId>
#     <version>2021.4.0</version>
# </dependency>