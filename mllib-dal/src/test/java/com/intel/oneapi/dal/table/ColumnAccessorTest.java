package com.intel.oneapi.dal.table;

import org.junit.jupiter.api.Test;

import static com.intel.oneapi.dal.table.Common.DataLayout.ROW_MAJOR;
import static org.junit.jupiter.api.Assertions.assertEquals;


public class ColumnAccessorTest {
    @Test
    public void getDoubleFirstColumnFromHomogenTable() throws Exception {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getDevice());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct(), CommonTest.getDevice());
        double[] columnData = accessor.pullDouble(0);
        assertEquals(new Long(columnData.length), table.getRowCount());
        double[] tableData = table.getDoubleData();
        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[i * table.getColumnCount().intValue()]);
        }
    }

    @Test
    public void getDoubleSecondColumnFromHomogenTableWithConversion() throws Exception {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getDevice());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct(), CommonTest.getDevice());
        double[] columnData = accessor.pullDouble(1);
        assertEquals(new Long(columnData.length), table.getRowCount());
        double[] tableData = table.getDoubleData();

        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[i * table.getColumnCount().intValue() + 1]);
        }
    }

    @Test
    public void getDoubleSecondColumnFromHomogenTableWithSubsetOfRows() throws Exception {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getDevice());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct(), CommonTest.getDevice());
        double[] columnData = accessor.pullDouble(0, 1 , 3);

        assertEquals(new Long(columnData.length), new Long(2));
        double[] tableData = table.getDoubleData();
        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[2 + i * table.getColumnCount().intValue()]);
        }
    }

    @Test
    public void getFloatFirstColumnFromHomogenTable() throws Exception {
        float[] data = {5.236359f, 8.718667f, 40.724176f, 10.770023f, 90.119887f, 3.815366f,
                53.620204f, 33.219769f, 85.208661f, 15.966239f};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getDevice());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct(), CommonTest.getDevice());
        float[] columnData = accessor.pullFloat(0);

        assertEquals(new Long(columnData.length), table.getRowCount());
        float[] tableData = table.getFloatData();
        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[i * table.getColumnCount().intValue()]);
        }
    }

    @Test
    public void getFloatSecondColumnFromHomogenTableWithConversion() throws Exception {
        float[] data = {5.236359f, 8.718667f, 40.724176f, 10.770023f, 90.119887f, 3.815366f,
                53.620204f, 33.219769f, 85.208661f, 15.966239f};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getDevice());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct(), CommonTest.getDevice());
        float[] columnData = accessor.pullFloat(1);

        assertEquals(new Long(columnData.length), table.getRowCount());
        float[] tableData = table.getFloatData();

        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[i * table.getColumnCount().intValue() + 1]);
        }
    }

    @Test
    public void getFloatSecondColumnFromHomogenTableWithSubsetOfRows() throws Exception {
        float[] data = {5.236359f, 8.718667f, 40.724176f, 10.770023f, 90.119887f, 3.815366f,
                53.620204f, 33.219769f, 85.208661f, 15.966239f};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getDevice());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct(), CommonTest.getDevice());
        float[] columnData = accessor.pullFloat(0, 1 , 3);

        assertEquals(new Long(columnData.length), new Long(2));
        float[] tableData = table.getFloatData();
        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[2 + i * table.getColumnCount().intValue()]);
        }
    }

    @Test
    public void getIntFirstColumnFromHomogenTable() throws Exception {
        int[] data = {5, 8, 40, 10, 90, 3, 53, 33, 85, 15};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getDevice());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct(), CommonTest.getDevice());
        int[] columnData = accessor.pullInt(0);
        assertEquals(new Long(columnData.length), table.getRowCount());
        int[] tableData = table.getIntData();
        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[i * table.getColumnCount().intValue()]);
        }
    }

    @Test
    public void getIntSecondColumnFromHomogenTableWithConversion() throws Exception {
        int[] data = {5, 8, 40, 10, 90, 3, 53, 33, 85, 15};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getDevice());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct(), CommonTest.getDevice());
        int[] columnData = accessor.pullInt(1);

        assertEquals(new Long(columnData.length), table.getRowCount());
        int[] tableData = table.getIntData();

        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[i * table.getColumnCount().intValue() + 1]);
        }
    }

    @Test
    public void getIntSecondColumnFromHomogenTableWithSubsetOfRows() throws Exception {
        int[] data = {5, 8, 40, 10, 90, 3, 53, 33, 85, 15};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getDevice());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct(), CommonTest.getDevice());
        int[] columnData = accessor.pullInt(0, 1 , 3);
        assertEquals(new Long(columnData.length), new Long(2));
        int[] tableData = table.getIntData();
        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[2 + i * table.getColumnCount().intValue()]);
        }
    }
}
