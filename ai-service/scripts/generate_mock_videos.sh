#!/usr/bin/env bash
# scripts/generate_mock_videos.sh
# ─────────────────────────────────────────────────────────────────────────────
# Generates 6 warehouse-style mock camera feeds using FFmpeg.
#
# Each feed is a unique color-coded video with:
#   • Camera ID watermark
#   • Zone label
#   • Timestamp overlay
#   • Simulated "scanline" noise
#   • Random motion (lavfi noise filter)
#
# Output: mock_sources/camera_01.mp4 … camera_06.mp4
#
# Usage:
#   chmod +x scripts/generate_mock_videos.sh
#   ./scripts/generate_mock_videos.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
OUTPUT_DIR="./mock_sources"
mkdir -p "$OUTPUT_DIR"

DURATION=30          # seconds per video (loops continuously at runtime)
FPS=10
WIDTH=640
HEIGHT=360

# Camera configs: ID, hex color, zone label
declare -a CAMERAS=(
  "camera_01 0x1a2744 'MAIN GATE / ENTRY ZONE'"
  "camera_02 0x0d2218 'WAREHOUSE AISLE / STORAGE'"
  "camera_03 0x1f1a08 'LOADING DOCK / ZONE'"
  "camera_04 0x0d1a26 'RACK SECTION / STORAGE'"
  "camera_05 0x2a0d0d 'RESTRICTED AREA'"
  "camera_06 0x1a0d26 'PACKING / DISPATCH AREA'"
)

echo "Generating $((${#CAMERAS[@]})) mock camera feeds..."
echo "Output directory: $OUTPUT_DIR"
echo ""

for entry in "${CAMERAS[@]}"; do
  # Parse: filename color "label"
  read -r filename color label <<< "$entry"
  output="$OUTPUT_DIR/${filename}.mp4"
  camid="${filename/camera_0/cam-0}"

  echo "  → $filename  ($label)"

  ffmpeg -y \
    -f lavfi \
    -i "color=c=${color}:size=${WIDTH}x${HEIGHT}:rate=${FPS},noise=alls=8:allf=t+u" \
    -t "$DURATION" \
    -vf "
      drawtext=text='${camid}':
        fontcolor=0x00e5ff:fontsize=14:x=10:y=10:
        font='monospace',
      drawtext=text='${label}':
        fontcolor=0x8892a4:fontsize=10:x=10:y=30:
        font='monospace',
      drawtext=text='%{localtime\:%Y-%m-%d %T}':
        fontcolor=0x00e5ff88:fontsize=10:
        x=10:y=${HEIGHT}-20:font='monospace',
      drawgrid=width=80:height=60:thickness=1:color=0x00e5ff08
    " \
    -c:v libx264 \
    -preset ultrafast \
    -crf 28 \
    -pix_fmt yuv420p \
    -an \
    "$output" 2>/dev/null

  echo "     ✅ $output ($(du -sh "$output" | cut -f1))"
done

echo ""
echo "✅ All mock videos generated in $OUTPUT_DIR"
echo ""
echo "Now run the AI service:"
echo "  cd warehouse-ai-service"
echo "  python main.py"
