#!/bin/bash
# camera_doctor.sh -- why won't the ZED X cameras open, and what actually fixes it?
#
# A camera that refuses to open always prints the same thing from the ZED SDK
# ("CAMERA STREAM FAILED TO START"), whatever the cause. The cause is only
# visible in nvargus-daemon's log. This reads both and names the case.
#
#   ./camera_doctor.sh          diagnose
#   ./camera_doctor.sh --stop   stop every camera client GRACEFULLY, then diagnose
#
# Cases, cheapest fix first (all four were hit on the car 2026-09-15/16):
#
#   HELD        another process has the camera (one client per ZED X).
#               -> stop that process. No restart helps while it holds it.
#   LINK        link dropping mid-stream ("CAMERA REBOOTING", "Connection
#               issue"). Usually a marginal GMSL connector, worse with
#               vibration outdoors.  -> reseat the cable.
#   GL          a camera opened while something held the display's GL context
#               ("BadParameter ... createBuffer", container exit -11).
#               -> start RViz AFTER the cameras, never before.
#   WEDGED      Argus itself is unhappy ("Cannot create camera provider",
#               "NvPclOpen: PCL Open Failed", "NOT DETECTED" with the serial
#               present in dmesg).  -> reboot. Do NOT restart nvargus-daemon
#               or zed_x_daemon: on 2026-09-16 that turned a recoverable
#               state into a crash loop on all three cameras.
set -u
SINCE=${SINCE:--10 min}

stop_clients() {
  echo "== stopping camera clients gracefully (SIGINT, then wait)"
  local pids
  pids=$(pgrep -f 'zed_camera.launch.py|component_container_isolated' || true)
  [ -z "$pids" ] && { echo "   none running"; return; }
  kill -INT $pids 2>/dev/null
  for _ in $(seq 20); do
    sleep 1
    pgrep -f 'zed_camera.launch.py|component_container_isolated' >/dev/null || break
  done
  if pgrep -f 'zed_camera.launch.py|component_container_isolated' >/dev/null; then
    echo "   still alive after 20 s; SIGKILL (the SDK will not have released the cameras)"
    pkill -9 -f 'zed_camera.launch.py|component_container_isolated'
  else
    echo "   all clients exited cleanly"
  fi
  echo "   waiting 15 s for Argus to release the sensors"; sleep 15
}

[ "${1:-}" = "--stop" ] && stop_clients

echo "== cameras streaming now"
for c in front left right; do
  n=$(pgrep -fc "__ns:=/zed_$c" || true)
  printf "   zed_%-6s driver=%s\n" "$c" "$([ "$n" -gt 0 ] && echo up || echo down)"
done

echo "== sensors the kernel sees (expect each serial twice: two imagers per camera)"
sudo -n dmesg 2>/dev/null | grep -a 'zedx_probe: Serial' | awk '{print "   "$NF}' | sort | uniq -c \
  || echo "   (needs sudo)"

echo "== nvargus-daemon, last $SINCE"
ARGUS=$(sudo -n journalctl -u nvargus-daemon --since "$SINCE" 2>/dev/null | tail -200)
echo "$ARGUS" | grep -aoE 'AlreadyAllocated|Cannot create camera provider|NvPclOpen: PCL Open Failed|Sensor could not be opened|BadParameter' \
  | sort | uniq -c | sed 's/^/   /' || echo "   (nothing)"

echo "== ZED wrapper logs, last lines"
for c in front left right; do
  f=/tmp/zed_$c.log
  [ -r "$f" ] || continue
  printf "   zed_%-6s failures=%s relaunches=%s\n" "$c" \
    "$(grep -ac 'FAILED TO START\|NOT DETECTED\|BadParameter' "$f")" "$(grep -ac watchdog "$f")"
done

echo "== verdict"
if echo "$ARGUS" | grep -q AlreadyAllocated; then
  echo "   HELD -- another process owns a camera. Find it:"
  echo "     ps -eo pid,args | grep -iE 'zed|camera' | grep -v component_container"
  echo "   Stop that process; no restart or reboot helps while it holds the camera."
elif tail -c 20000 /tmp/zed_*.log 2>/dev/null | grep -q 'BadParameter'; then
  echo "   GL -- a camera opened while something held the display's GL context."
  echo "   Start RViz only AFTER the cameras (perception_stack.launch.py does this)."
  echo "   Recover: ./camera_doctor.sh --stop, then start the cameras again."
elif echo "$ARGUS" | grep -qE 'Cannot create camera provider|NvPclOpen|Sensor could not be opened'; then
  echo "   WEDGED -- Argus is not serving cameras. Reboot."
  echo "   Do NOT restart nvargus-daemon / zed_x_daemon; that made it worse."
elif tail -c 20000 /tmp/zed_*.log 2>/dev/null | grep -qE 'CAMERA REBOOTING|Connection issue'; then
  echo "   LINK -- the GMSL link is dropping mid-stream. Reseat that camera's cable"
  echo "   at both ends; it gets worse with vibration outdoors."
else
  echo "   No known failure signature in the last $SINCE."
fi
