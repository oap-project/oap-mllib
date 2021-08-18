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

package org.apache.spark.ml.util;

import com.intel.daal.utils.LibUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.UUID;

public final class LibLoader {
  private static final String LIBRARY_PATH_IN_JAR = "/lib";
  // Make sure loading libraries from different temp directory for each process
  private static final String subDir = "MLlibDAL_" + UUID.randomUUID();

  private static final Logger log = LoggerFactory.getLogger("LibLoader");

  private static boolean isLoaded = false;

  /**
   * Get temp dir for exacting lib files
   *
   * @return path of temp dir
   */
  public static String getTempSubDir() {
    String tempSubDirectory = System.getProperty("java.io.tmpdir") +
            "/" + subDir + LIBRARY_PATH_IN_JAR;
    return tempSubDirectory;
  }

  /**
   * Load all native libs
   */
  public static synchronized void loadLibraries() throws IOException {
    if (isLoaded)
      return;

    if (!loadLibSYCL()) {
      log.debug("SYCL libraries are not available, will load CPU libraries only.");
    }
    loadLibCCL();
    loadLibMLlibDAL();

    isLoaded = true;
  }

  /**
   * Load oneCCL libs in dependency order
   */
  private static synchronized void loadLibCCL() throws IOException {
    // Load libfabric from system first, if failed load from jar
    if (!loadFromSystem("libfabric.so.1")) {
      loadFromJar(subDir, "libfabric.so.1");
      loadFromJar(subDir, "libsockets-fi.so");
    }
    loadFromJar(subDir, "libmpi.so.12");
    loadFromJar(subDir, "libccl.so");
  }

  /**
   * Load SYCL libs in dependency order
   */
  private static synchronized Boolean loadLibSYCL() throws IOException {
    // Check if SYCL libraries are available
    InputStream streamIn = LibLoader.class.getResourceAsStream(LIBRARY_PATH_IN_JAR + "/libsycl.so.5");
    if (streamIn == null) {
      return false;
    }
    streamIn.close();

    loadFromJar(subDir, "libintlc.so.5");
    loadFromJar(subDir, "libimf.so");
    loadFromJar(subDir, "libirng.so");
    loadFromJar(subDir, "libsvml.so");
    loadFromJar(subDir, "libOpenCL.so.1");
    loadFromJar(subDir, "libsycl.so.5");

    return true;
  }

  /**
   * Load MLlibDAL lib
   */
  private static synchronized void loadLibMLlibDAL() throws IOException {
    // oneDAL Java API doesn't load correct libtbb version
    // See https://github.com/oneapi-src/oneDAL/issues/1254
    // Workaround: Load packaged libtbb & libtbbmalloc & libJavaAPI.so manually

    // Load TBB & oneDAL JavaAPI libs
    loadFromJar(subDir, "libtbb.so.12");
    loadFromJar(subDir, "libtbbmalloc.so.2");
    loadFromJar(subDir, "libJavaAPI.so");

    loadFromJar(subDir, "libMLlibDAL.so");
  }

  private static synchronized boolean loadFromSystem(String name) {
    try {
      System.loadLibrary(name);
    } catch (UnsatisfiedLinkError e) {
      return false;
    }
    return true;
  }

  /**
   * Load lib as resource
   *
   * @param path sub folder (in temporary folder) name
   * @param name library name
   */
  private static void loadFromJar(String path, String name) throws IOException {
    log.debug("Loading " + name + " ...");

    File fileOut = createTempFile(path, name);
    // File exists already
    if (fileOut == null) {
      log.debug("DONE: Loading library as resource.");
      return;
    }

    InputStream streamIn = LibLoader.class.getResourceAsStream(LIBRARY_PATH_IN_JAR + "/" + name);
    if (streamIn == null) {
      throw new IOException("Error: No resource found.");
    }

    try (OutputStream streamOut = new FileOutputStream(fileOut)) {
      log.debug("Writing resource to temp file.");

      byte[] buffer = new byte[32768];
      while (true) {
        int read = streamIn.read(buffer);
        if (read < 0) {
          break;
        }
        streamOut.write(buffer, 0, read);
      }

      streamOut.flush();
    } catch (IOException e) {
      throw new IOException("Error:  I/O error occurs from/to temp file.");
    } finally {
      streamIn.close();
    }

    System.load(fileOut.toString());
    log.debug("DONE: Loading library as resource.");
  }

  /**
   * Create temporary file
   *
   * @param name           library name
   * @param tempSubDirName sub folder (in temporary folder) name
   * @return temporary file handler. null if file exist already.
   */
  private static File createTempFile(String tempSubDirName, String name) throws IOException {
    File tempSubDirectory = new File(
            System.getProperty("java.io.tmpdir") + "/" + tempSubDirName + LIBRARY_PATH_IN_JAR);

    if (!tempSubDirectory.exists()) {
      tempSubDirectory.mkdirs();
      // Check existance again, don't use return bool of mkdirs
      if (!tempSubDirectory.exists()) {
        throw new IOException("Error: Can`t create folder for temp file.");
      }
    }

    String tempFileName = tempSubDirectory + "/" + name;
    File tempFile = new File(tempFileName);

    if (tempFile == null) {
      throw new IOException("Error: Can`t create temp file.");
    }

    if (tempFile.exists()) {
      return null;
    }

    return tempFile;
  }

}
