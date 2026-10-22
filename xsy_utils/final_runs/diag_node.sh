#!/usr/bin/env bash
# Snapshot of a (possibly hung) training node, written to CPFS so it can be read from the dev box.
# Never blocks on a stuck rank: py-spy runs --nonblocking under a timeout, everything else reads /proc.
# Usage (inside the DLC container, e.g. web terminal):  bash diag_node.sh [out_dir]
# Also called by train.sh's hang watchdog (copied from the legacy lingbot-va2 DLC setup).
OUT_DIR=${1:-/mnt/cpfs/linbot-vla-v2/.runtime/logs/diag}
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/diag_$(hostname)_$(date +%m%d_%H%M%S).txt"
PYSPY=${PYSPY:-/mnt/cpfs/venvs/lingbotvla/bin/py-spy}
sec() { echo; echo "===== $*"; }
{
    sec "date / uptime / nproc"; date; uptime; nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null
    sec "free -m"; free -m
    sec "/proc/meminfo"; cat /proc/meminfo
    sec "memory pressure"; cat /proc/pressure/memory /proc/pressure/io 2>/dev/null
    sec "cgroup memory"; for f in memory.current memory.max memory.stat memory.events \
        memory/memory.usage_in_bytes memory/memory.limit_in_bytes memory/memory.stat; do
        [ -r /sys/fs/cgroup/$f ] && { echo "-- $f"; head -40 /sys/fs/cgroup/$f; }; done
    sec "df (tmpfs / shm / root / tmp)"; df -h / /tmp /dev/shm /run 2>/dev/null; df -h -t tmpfs 2>/dev/null
    sec "du /dev/shm /tmp /root (top)"; du -xsh /dev/shm/* /tmp/* /root/.cache/* 2>/dev/null | sort -h | tail -15
    sec "procs by RSS (rss/pss MB, state, wchan)"
    ps -eo pid,ppid,stat,wchan:32,rss,etime,args --sort=-rss | head -60 | cut -c1-260
    sec "D-state procs + kernel stack"
    for p in $(ps -eo pid,stat | awk '$2 ~ /D/ {print $1}' | head -20); do
        echo "-- pid $p $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-160)"
        cat /proc/$p/wchan 2>/dev/null; echo; timeout 5 cat /proc/$p/stack 2>/dev/null
    done
    sec "per-rank smaps_rollup (Rss/Pss/Shared/Anon)"
    for p in $(pgrep -f 'tasks/vla/train_lingbotvla.py'); do
        echo "-- pid $p ppid $(ps -o ppid= -p $p) $(grep -E '^(Rss|Pss|Pss_Anon|Pss_File|Pss_Shmem|Shared_Clean|Shared_Dirty|Anonymous):' /proc/$p/smaps_rollup 2>/dev/null | awk '{printf "%s%dM ", $1, $2/1024}')"
    done | head -120
    sec "nvidia-smi"; timeout 30 nvidia-smi
    sec "py-spy (nonblocking) of training ranks"
    for tr in $(pgrep -f 'torch.distributed.run'); do
        for r in $(pgrep -P "$tr"); do
            echo "----- rank pid $r"; timeout 60 "$PYSPY" dump --nonblocking --pid "$r" 2>&1 | head -80
        done
    done
    sec "dmesg tail"; dmesg -T 2>/dev/null | tail -40
} > "$OUT" 2>&1
echo "$OUT"
