package com.intel.oneapi.dal.table;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;


public class RowAccessorTest {
    @Test
    public void readDoubleTableDataFromRowAccessor() {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getComputeDevice());

        RowAccessor accessor = new RowAccessor(table.getcObejct(), CommonTest.getComputeDevice());
        double[] rowData = accessor.pullDouble(0, table.getRowCount());
        assertEquals(new Long(rowData.length),
                new Long(table.getColumnCount() * table.getRowCount()));
        assertArrayEquals(rowData, data);
        for (int i = 0; i < rowData.length; i++) {
            assertEquals(rowData[i], data[i]);
        }
    }

    @Test
    public void readFloatTableDataFromRowAccessor() {
        float[] data = {5.236359f, 8.718667f, 40.724176f, 10.770023f, 90.119887f, 3.815366f,
                53.620204f, 33.219769f, 85.208661f, 15.966239f};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getComputeDevice());

        RowAccessor accessor = new RowAccessor(table.getcObejct(), CommonTest.getComputeDevice());
        float[] rowData = accessor.pullFloat(0, table.getRowCount());
        assertEquals(new Long(rowData.length),
                new Long(table.getColumnCount() * table.getRowCount()));
        assertArrayEquals(rowData, data);
        for (int i = 0; i < rowData.length; i++) {
            assertEquals(rowData[i], data[i]);
        }
    }

    @Test
    public void readIntTableDataFromRowAccessor() {
        int[] data = {5, 8, 40, 10, 90, 3, 53, 33, 85, 15};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getComputeDevice());

        RowAccessor accessor = new RowAccessor(table.getcObejct(),
                CommonTest.getComputeDevice());
        int[] rowData = accessor.pullInt(0, table.getRowCount());
        assertEquals(new Long(rowData.length),
                new Long(table.getColumnCount() * table.getRowCount()));
        assertArrayEquals(rowData, data);
        for (int i = 0; i < rowData.length; i++) {
            assertEquals(rowData[i], data[i]);
        }
    }
}
