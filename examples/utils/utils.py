from datetime import datetime
import csv

class Timer():

    def __init__(self, algoName):

        self._algoName = algoName
        self._filename = algoName+"1time_breakdown"
        f = open(self._filename, "w")
        f.write("Stage,Stamp\n")
        f.close()

    def record(self, time_stamp_name):
        time_str = str(datetime.now())[:-3]
        f = open(self._filename, "a")
        f.write(time_stamp_name + "," + time_str+"\n")
        f.close()
    
    def printTimeTable(self):
        time_metrics = []
        with open(self._filename, "r") as timeFile:
            time_reader = csv.DictReader(timeFile, delimiter=",")
            for entry in time_reader:
                print(entry)
                stage_name = entry["Stage"]
                time_stamp = entry["Stamp"]
                time_metrics.append([stage_name, datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S.%f")])
        time_interval = [["AlgoName", time_metrics[0][0]], [self._algoName, str(time_metrics[0][1])]]
        pre = 0
        for i in range(len(time_metrics)):
            if i == 0:
                continue
            else:
                time_interval[0].append(time_metrics[i][0])
                time_interval[1].append(str(time_metrics[i][1] - time_metrics[i-1][1])[:-3])
        time_interval[0].append("Total")
        time_interval[1].append(str(time_metrics[len(time_metrics) - 1][1]-time_metrics[0][1])[:-3])

        format_row = "{:>20}" + "{:>30}" * (len(time_interval[0]) - 1)
        print(format_row.format(*time_interval[0]))
        print(format_row.format(*time_interval[1]))
