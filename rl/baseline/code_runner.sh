#!/bin/bash

GENERATOR_OPT="simple"
BIAS_OPT="min"
PROPERTY_OPT="logP"

NUM_RUNS=2
I=1

while [ $I -le $NUM_RUNS ]
do 
   python RL_baseline_trimmed.py $GENERATOR_OPT $BIAS_OPT $PROPERTY_OPT $I
   ((I++))
done

python eval_strings_trimmed.py $GENERATOR_OPT $BIAS_OPT $PROPERTY_OPT $NUM_RUNS > one_sample_stats/"$GENERATOR_OPT"_"$BIAS_OPT"_"$PROPERTY_OPT"-stats.txt
