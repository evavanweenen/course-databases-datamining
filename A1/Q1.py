import numpy as np
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
"""
select      l_returnflag,
            l_linestatus,
            sum(l_quantity) as sum_qty,
            sum(l_extendedprice) as sum_base_price,
            sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
            sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
            avg(l_quantity) as avg_qty,
            avg(l_extendedprice) as avg_price,
            avg(l_discount) as avg_disc,
            count(*) as count_order
from        lineitem
where       l_shipdate <= date '1998-12-01' - interval '90' day (3)
group by    l_returnflag, l_linestatus
order by    l_returnflag, l_linestatus;
"""


#implement "where" statement
selection = [(time.strptime(i, "%Y-%m-%d") <= (time.strptime('1998-09-02',"%Y-%m-%d"))) for i in data['l_shipdate'].values]

#implement "select" statements by group
sum_qty = data['l_quantity'][selection].groupby([data['l_returnflag'], data['l_linestatus']]).sum()
sum_base_price = data['l_extendedprice'][selection].groupby([data['l_returnflag'], data['l_linestatus']]).sum()

disc_price = data['l_extendedprice'][selection]*(1-data['l_discount'][selection])
sum_disc_price = disc_price.groupby([data['l_returnflag'], data['l_linestatus']]).sum()

charge = data['l_extendedprice'][selection]*(1-data['l_discount'][selection])*(1 + data['l_tax'][selection])
sum_charge = charge.groupby([data['l_returnflag'], data['l_linestatus']]).sum()

avg_qty = data['l_quantity'][selection].groupby([data['l_returnflag'], data['l_linestatus']]).mean()
avg_price = data['l_extendedprice'][selection].groupby([data['l_returnflag'], data['l_linestatus']]).mean()
avg_disc = data['l_discount'][selection].groupby([data['l_returnflag'], data['l_linestatus']]).mean()

count_order = data[selection].groupby([data['l_returnflag'], data['l_linestatus']]).size()

#order groups and put into 1 table

output=pd.DataFrame(data=([sum_qty, sum_base_price, sum_disc_price,sum_charge,avg_qty,avg_price,avg_disc,count_order]) ).T
output.columns=['sum_qty', 'sum_base_price', 'sum_disc_price','sum_charge','avg_qty','avg_price','avg_disc','count_order']
output.sort_values(by=['l_returnflag', 'l_linestatus'])

print(output)

end_wallclock = time.time()
end_abscpu = time.perf_counter()
end_cpu = time.process_time()
print("Elapsed wall-clock time = " , str(end_wallclock-start_wallclock), ' seconds')
print("Elapsed absolute cpu time = " , str(end_abscpu-start_abscpu), ' seconds')
print("Elapsed (user + system) cpu time = " , str(end_cpu-start_cpu), ' seconds')
