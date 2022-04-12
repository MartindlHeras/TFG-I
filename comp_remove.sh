#!/bin/bash
echo "Removing comp data!\n"
sleep 1


echo "Counting the number of mutants\n"
cd mutants
count=$(ls -l |wc -l)
echo $count
sleep 1

# save the current time
start_time=$( date +%s.%N )


cd ..
echo "Removing mutants: "
for i in `seq 0 $count`;
do	echo " ======================> "$i
        rm mutants/$i/massive
done

# the current time after the program has finished
# minus the time when we started, in seconds.nanoseconds
elapsed_time=$( date +%s.%N --date="$start_time seconds ago" )

echo elapsed_time: $elapsed_time
