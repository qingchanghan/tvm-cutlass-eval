First run `export.py` to dump bert model. The logs is tuned for A10, but you can easily tune for other cards.

+ `ansor.py`: Auto-Scheduler
+ `cutlass.py`: CUTLASS
+ `cutlass_ansor.py`: CUTLASS + Ansor
+ `meta_schedule.py`: meta schedule
+ `cutlass_ms.py`: CUTLASS + MetaSchedule


### End2end latency(ms) on A10

| Ansor (n=3000)   | CUTLASS+TOPI | CUTLASS+Ansor (n=3000) | MetaSchedule (n=3000) | CUTLASS+MetaSchedule (n=3000) |
| ------- | ------------ | ------------- | ------------- | ------------- |
| 55.8870 | 20.2297      | 17.2543       | 19.2774       | 17.5876       |
