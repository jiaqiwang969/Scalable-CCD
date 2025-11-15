#!/usr/bin/env bash
# 下载 NYU Archive 数据集到本地并解压到目标目录。
# 用法:
#   scripts/fetch_full_dataset.sh "https://archive.nyu.edu/handle/2451/74508?mode=full" tests/data-full
#
# 说明:
# - 需要本机可访问外网。
# - 会在目标目录下创建 _src 存放原始压缩包，解压内容直接落到目标目录。
# - 解析页面中的 bitstream 链接并批量下载；使用 --content-disposition 保留服务器提供的文件名。

set -euo pipefail

URL="${1:-}"
DEST="${2:-}"
if [[ -z "${URL}" || -z "${DEST}" ]]; then
  echo "用法: $0 <NYU_HANDLE_URL> <DEST_DIR>" >&2
  exit 1
fi

echo "[fetch] 目标目录: ${DEST}"
mkdir -p "${DEST}/_src"
SRC="${DEST}/_src"

fetch_cmd=""
if command -v wget >/dev/null 2>&1; then
  fetch_cmd="wget -q"
elif command -v curl >/dev/null 2>&1; then
  fetch_cmd="curl -sSL -o"
else
  echo "[fetch] 需要 wget 或 curl" >&2
  exit 2
fi

BASE="https://archive.nyu.edu"
PAGE="${SRC}/nyu_page.html"

echo "[fetch] 抓取页面: ${URL}"
if [[ "${fetch_cmd}" == wget* ]]; then
  wget -q -O "${PAGE}" "${URL}"
else
  curl -sSL "${URL}" -o "${PAGE}"
fi
echo "[fetch] 页面长度: $(wc -c < "${PAGE}") 字节"

echo "[fetch] 解析 bitstream 链接..."
LINKS_RAW="${SRC}/links_raw.txt"
LINKS="${SRC}/links.txt"
grep -oE 'href=\"[^\"]*bitstream[^\"]*\"' "${PAGE}" \
  | sed -E 's/^href=\"(.*)\"$/\1/' \
  | sort -u > "${LINKS_RAW}"

# 归一化为绝对URL，并尽量追加 download 参数
> "${LINKS}"
while IFS= read -r L; do
  # 前缀补全
  if [[ "${L}" =~ ^/ ]]; then
    L="${BASE}${L}"
  elif [[ ! "${L}" =~ ^https?:// ]]; then
    # 相对链接：拼接基地址
    L="${BASE}/${L}"
  fi
  # 优先确保带 download 参数
  if [[ "${L}" =~ /download($|\?) ]]; then
    : # 已经是下载地址
  elif [[ "${L}" =~ \? ]]; then
    L="${L}&download=1"
  else
    L="${L}?download=1"
  fi
  echo "${L}" >> "${LINKS}"
done < "${LINKS_RAW}"
sort -u -o "${LINKS}" "${LINKS}"

echo "[fetch] 发现链接数量: $(wc -l < "${LINKS}")"
if [[ ! -s "${LINKS}" ]]; then
  echo "[fetch] 未解析到任何可下载链接，请检查页面结构或手动获取直链。" >&2
  exit 3
fi

echo "[fetch] 开始批量下载到 ${SRC}"
while IFS= read -r U; do
  echo "  - ${U}"
  # 使用 --content-disposition 以保留服务器提供的文件名
  wget -c --content-disposition -P "${SRC}" "${U}" || {
    echo "[warn] 下载失败: ${U}"
  }
done < "${LINKS}"

echo "[fetch] 下载完成，开始解压..."
shopt -s nullglob
for F in "${SRC}"/*; do
  NAME="$(basename -- "${F}")"
  LOWER="${NAME,,}"
  case "${LOWER}" in
    *.zip)
      echo "  - 解压 zip: ${NAME}"
      unzip -q -n "${F}" -d "${DEST}"
      ;;
    *.tar.gz|*.tgz)
      echo "  - 解压 tar.gz: ${NAME}"
      tar -xzf "${F}" -C "${DEST}"
      ;;
    *.tar)
      echo "  - 解压 tar: ${NAME}"
      tar -xf "${F}" -C "${DEST}"
      ;;
    *.7z)
      if command -v 7z >/dev/null 2>&1; then
        echo "  - 解压 7z: ${NAME}"
        7z x -y "-o${DEST}" "${F}" >/dev/null
      else
        echo "[warn] 跳过 7z（未安装 7z）: ${NAME}"
      fi
      ;;
    *.gz)
      # 处理单文件 .gz
      echo "  - 解压 gz: ${NAME}"
      gunzip -c "${F}" > "${DEST}/${NAME%.gz}" || echo "[warn] 解压失败: ${NAME}"
      ;;
    *)
      # 保留未知类型原样
      echo "  - 保留文件（未知类型）: ${NAME}"
      mkdir -p "${DEST}/_loose"
      cp -n "${F}" "${DEST}/_loose/${NAME}" || true
      ;;
  esac
done
shopt -u nullglob

echo "[fetch] 完成。解压目标: ${DEST}"
echo "[fetch] 如果目录结构与期望不同，请告诉我们，我们会补充自动适配逻辑。"

