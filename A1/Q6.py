import pandas as pd
import time

start_wallclock = time.time() #returns a real-world time
start_abscpu = time.perf_counter() #returns the absolute value of the cpu time (so keeps counting when system goes to sleep)
start_cpu = time.process_time() #returns the cpu time counter, but is only updated when a given process is running on the CPU (this is system-time and user-time combined)
#REFERENCE: https://stackoverflow.com/questions/25785243/understanding-time-perf-counter-and-time-process-time
#NOTE: don't use time.clock(), this is deprecated after a certain version

table = '/disks/strw13/DBDM/tpch_2_17_1/dbgen/SF-3/data/lineitem.tbl'

names = ('l_orderkey', 'l_linenumber', 'l_partkey', 'l_suppkey', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstructions', 'l_shipmode', 'l_comment')

data = pd.read_table(table, sep='|', header=None, names=names, index_col=False)

#implement "where" statements
shipdate = [(time.strptime(i, "%Y-%m-%d")>= time.strptime('1994-01-01',"%Y-%m-%d")) & (time.strptime(i, "%Y-%m-%d")< time.strptime('1995-01-01',"%Y-%m-%d")) for i in data['l_shipdate'].values]

discount = [(0.05 <= data['l_discount'].values) & (0.07 >= data['l_discount'].values)]

quantity = [data['l_quantity'].values < 24]

#select data for which all "where" statements are True
selection = [(shipdate) & (discount[0]) & (quantity[0])]

#Calculate Revenue
revenue = sum(data['l_extendedprice'][selection[0]] * data['l_discount'][selection[0]])

end_wallclock = time.time()
end_abscpu = time.perf_counter()
end_cpu = time.process_time()

print("Revenue: ", revenue)
print("Elapsed wall-clock time = " , str(end_wallclock-start_wallclock), ' seconds')
print("Elapsed absolute cpu time = " , str(end_abscpu-start_abscpu), ' seconds')
print("Elapsed (user + system) cpu time = " , str(end_cpu-start_cpu), ' seconds')
