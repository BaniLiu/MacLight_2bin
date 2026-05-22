param(
    [string]$Python = "C:\Users\pliu0073\.conda\envs\2bin_DRL_TSC\python.exe",
    [string]$Task = "block",
    [string]$Level = "normal",
    [int]$BlockNum = 8,
    [int]$Seconds = 3600,
    [int]$Episodes = 80,
    [int]$SeedStart = 42,
    [int]$SeedEnd = 46
)

$ErrorActionPreference = "Stop"

& $Python run_Fixed.py -w 1 -t $Task -l $Level -b $BlockNum --seconds $Seconds -e $Episodes -s $SeedStart $SeedEnd
& $Python run_IPPO.py -w 1 -t $Task -l $Level -b $BlockNum --seconds $Seconds -e $Episodes -s $SeedStart $SeedEnd
& $Python run_IDQN.py -w 1 -t $Task -l $Level -b $BlockNum --seconds $Seconds -e $Episodes -s $SeedStart $SeedEnd
& $Python compare_models.py --scene "$($Task)_$($Level)"
