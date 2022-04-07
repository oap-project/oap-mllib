package com.intel.oneapi.dal.table;


import org.junit.Test;

import static com.intel.oneapi.dal.table.Common.DataLayout.column_major;
import static com.intel.oneapi.dal.table.Common.DataLayout.row_major;
import static com.intel.oneapi.dal.table.Common.DataType.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class HomogenTableTest {
    @Test
    // can construct rowmajor int table 5x2
    public void create_rowmajor_int_table() throws Exception {
        int data[] = {1, 2, 3, 4, 5, 6, 10, 80, 10, 11};
        HomogenTable table = new HomogenTable(5, 2,
                data, int32, row_major);

        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());
        assertEquals(Common.DataLayout.row_major,table.getDataLayout());

        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            System.out.println(metadata.getFeatureCount());
            System.out.println(metadata.getDataType(i));
            assertEquals(metadata.getDataType(i), int32);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ordinal);
        }

        assertArrayEquals(data, table.getIntData());
    }
    @Test
    // can construct rowmajor double table 5x2
    public void create_rowmajor_double_table() throws Exception {
        double data[] = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, float64, row_major);

        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());
        assertEquals(Common.DataLayout.row_major,table.getDataLayout());

        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            System.out.println(metadata.getFeatureCount());
            System.out.println(metadata.getDataType(i));
            assertEquals(metadata.getDataType(i), float64);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ratio);
        }

        assertArrayEquals("", data, table.getDoubleData(), 0.000001d);
    }
    @Test
    // can construct rowmajor long table 5x2
    public void create_rowmajor_long_table() throws Exception {
        long data[] = {1l, 2l, 3l, 4l, 5l, 6l, 10l, 80l, 10l, 11l};
        HomogenTable table = new HomogenTable(5, 2,
                data, int64, row_major);

        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());
        assertEquals(Common.DataLayout.row_major,table.getDataLayout());

        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            System.out.println(metadata.getFeatureCount());
            System.out.println(metadata.getDataType(i));
            assertEquals(metadata.getDataType(i), int64);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ordinal);
        }

        assertArrayEquals(data, table.getLongData());
    }
    @Test
    // can construct rowmajor float table 5x2
    public void create_rowmajor_float_table() throws Exception {
        float data[] = {5.236359f, 8.718667f, 40.724176f, 10.770023f, 90.119887f, 3.815366f,
                53.620204f, 33.219769f, 85.208661f, 15.966239f};
        HomogenTable table = new HomogenTable(5, 2,
                data, float32, row_major);

        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());

        assertEquals(Common.DataLayout.row_major,table.getDataLayout());

        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), float32);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ratio);
        }

        assertArrayEquals("", data, table.getFloatData(), 0.000001f);
    }

    @Test
    // can construct colmajor int table
    public void create_colmajor_int_table() throws Exception {
        int data[] = {1, 2, 3, 4, 5, 6, 10, 80, 10, 11};
        HomogenTable table = new HomogenTable(5, 2,
                data, int32, column_major);

        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());

        assertEquals(column_major,table.getDataLayout());
        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), int32);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ordinal);
        }
        assertArrayEquals(data, table.getIntData());
    }

    @Test
    // can construct colmajor float table
    public void create_colmajor_float_table() throws Exception {
        float data[] = {5.236359f, 8.718667f, 40.724176f, 10.770023f, 90.119887f, 3.815366f,
                53.620204f, 33.219769f, 85.208661f, 15.966239f};
        HomogenTable table = new HomogenTable(5, 2,
                data, float32, column_major);

        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());

        assertEquals(column_major,table.getDataLayout());
        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), float32);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ratio);
        }
        assertArrayEquals("", data, table.getFloatData(), 0.000001f);
    }

    @Test
    // can construct colmajor long table
    public void create_colmajor_long_table() throws Exception {
        long data[] = {1l, 2l, 3l, 4l, 5l, 6l, 10l, 80l, 10l, 11l};
        HomogenTable table = new HomogenTable(5, 2,
                data, int64, column_major);

        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());

        assertEquals(column_major,table.getDataLayout());
        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), int64);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ordinal);
        }
        assertArrayEquals(data, table.getLongData());
    }

    @Test
    // can construct colmajor double table
    public void create_colmajor_double_table() throws Exception {
        double data[] = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, float64, column_major);

        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());

        assertEquals(column_major,table.getDataLayout());
        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), float64);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ratio);
        }
        assertArrayEquals("", data, table.getDoubleData(), 0.000001d);
    }
}
