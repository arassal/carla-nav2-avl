#!/bin/bash
# Minimal test-media fetch: COCO val street scenes (CC-licensed) + Wikimedia
# cones / grass-line images. Verifies each file decodes; logs sources.
set -u
mkdir -p "$(dirname "$0")/../testdata"
cd "$(dirname "$0")/../testdata" || exit 1
mkdir -p street cones grass_lines video
S=SOURCES.txt
: > $S
UA="drivingseg/0.1 (test media fetch)"

grab() {  # dir name url
  local f="$1/$2"
  curl -sL -A "$UA" --max-time 60 -o "$f" "$3" || return 1
  python3 - "$f" <<'EOF' || { rm -f "$f"; return 1; }
import sys, cv2
img = cv2.imread(sys.argv[1])
assert img is not None and img.size > 0
EOF
  echo "$f <- $3" >> $S
  echo "ok  $f"
}

# COCO val2017 street scenes (known ids with people/cars/lights/signs)
for id in 000000001296 000000002153 000000003501 000000011760 000000018380 \
          000000087038 000000174482 000000480985; do
  grab street "$id.jpg" "http://images.cocodataset.org/val2017/$id.jpg"
done

# Wikimedia: cones + chalk lines on grass (Special:FilePath resolves to file)
W="https://commons.wikimedia.org/wiki/Special:FilePath"
grab cones cone1.jpg "$W/Traffic%20cones%20on%20road.jpg?width=1280"
grab cones cone2.jpg "$W/Traffic_cone_on_a_street.jpg?width=1280"
grab cones cone3.jpg "$W/Orange_traffic_cones.jpg?width=1280"
grab cones cone4.jpg "$W/Traffic_cones_Autotest.jpg?width=1280"
grab grass_lines field1.jpg "$W/Soccer_field_-_empty.jpg?width=1280"
grab grass_lines field2.jpg "$W/Football_pitch_corner_flag.jpg?width=1280"
grab grass_lines field3.jpg "$W/White_line_on_grass.jpg?width=1280"

echo "---"; find . -type f \( -name '*.jpg' -o -name '*.mp4' \) | sort | xargs -r du -h
