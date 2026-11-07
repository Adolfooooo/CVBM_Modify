#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  bash clean_pth_recursive.sh [目录]                    # 只处理该目录，不递归，预览
  bash clean_pth_recursive.sh [目录] --delete           # 只处理该目录，真正删除

  bash clean_pth_recursive.sh [目录] --recursive        # 递归处理每个含 .pth 的目录，预览
  bash clean_pth_recursive.sh [目录] -r --delete        # 递归处理每个含 .pth 的目录，真正删除

  bash clean_pth_recursive.sh --dir /path/to/root -r
  bash clean_pth_recursive.sh --dir /path/to/root -r --delete

规则：
  对每个目录分别执行以下逻辑，不会把不同目录下的 pth 混在一起比较：
    1. 保留文件名中包含 best 的 .pth 文件
    2. 非 best 文件中：
       - 带 _ema_ 的 iter_*.pth 保留 iter 数字最大的最后 2 个
       - 不带 _ema_ 的 iter_*.pth 保留 iter 数字最大的最后 2 个
    3. 其余 .pth 文件删除
    4. 不符合 iter_数字_*.pth 命名规则的 .pth 文件默认保留，避免误删

说明：
  不传目录时，默认处理当前目录。
  默认是预览模式，不会真正删除；加 --delete 才会删除。
EOF
}

TARGET_DIR="."
DO_DELETE=0
RECURSIVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delete)
      DO_DELETE=1
      shift
      ;;
    --recursive|-r)
      RECURSIVE=1
      shift
      ;;
    --dir|-d)
      if [[ $# -lt 2 ]]; then
        echo "错误：$1 后面需要跟目录地址。" >&2
        exit 1
      fi
      TARGET_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      if [[ $# -gt 0 ]]; then
        TARGET_DIR="$1"
        shift
      fi
      ;;
    -*)
      echo "错误：未知参数 $1" >&2
      usage
      exit 1
      ;;
    *)
      TARGET_DIR="$1"
      shift
      ;;
  esac
done

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "错误：目录不存在：$TARGET_DIR" >&2
  exit 1
fi

TARGET_DIR="$(cd "$TARGET_DIR" && pwd -P)"

clean_one_dir() {
  local dir="$1"
  local tmpdir
  tmpdir="$(mktemp -d)"

  local ema_list="$tmpdir/ema.txt"
  local normal_list="$tmpdir/normal.txt"
  local keep_list="$tmpdir/keep.txt"
  local delete_list="$tmpdir/delete.txt"
  local unmatched_list="$tmpdir/unmatched.txt"

  : > "$ema_list"
  : > "$normal_list"
  : > "$keep_list"
  : > "$delete_list"
  : > "$unmatched_list"

  local oldpwd
  oldpwd="$(pwd -P)"
  cd "$dir"

  shopt -s nullglob
  local pth_files=( *.pth )

  if [[ ${#pth_files[@]} -eq 0 ]]; then
    cd "$oldpwd"
    rm -rf "$tmpdir"
    return 0
  fi

  local f base iter

  for f in "${pth_files[@]}"; do
    base="$(basename "$f")"

    if [[ "$base" == *best* ]]; then
      printf '%s\n' "$base" >> "$keep_list"
      continue
    fi

    if [[ "$base" =~ ^iter_([0-9]+)_.*\.pth$ ]]; then
      iter="${BASH_REMATCH[1]}"

      if [[ "$base" == *_ema_* ]]; then
        printf '%s\t%s\n' "$iter" "$base" >> "$ema_list"
      else
        printf '%s\t%s\n' "$iter" "$base" >> "$normal_list"
      fi
    else
      printf '%s\n' "$base" >> "$unmatched_list"
      printf '%s\n' "$base" >> "$keep_list"
    fi
  done

  if [[ -s "$normal_list" ]]; then
    sort -n -k1,1 "$normal_list" | tail -n 2 | cut -f2- >> "$keep_list"
  fi

  if [[ -s "$ema_list" ]]; then
    sort -n -k1,1 "$ema_list" | tail -n 2 | cut -f2- >> "$keep_list"
  fi

  sort -u "$keep_list" -o "$keep_list"

  for f in "${pth_files[@]}"; do
    base="$(basename "$f")"
    if ! grep -Fxq "$base" "$keep_list"; then
      printf '%s\n' "$base" >> "$delete_list"
    fi
  done

  echo
  echo "============================================================"
  echo "处理目录：$(pwd -P)"
  echo "============================================================"

  echo "========== 将保留的文件 =========="
  cat "$keep_list" || true

  echo
  echo "========== 将删除的文件 =========="
  if [[ -s "$delete_list" ]]; then
    cat "$delete_list"
  else
    echo "无"
  fi

  if [[ -s "$unmatched_list" ]]; then
    echo
    echo "========== 未匹配 iter_数字 命名、已默认保留的文件 =========="
    cat "$unmatched_list"
  fi

  if [[ "$DO_DELETE" -eq 1 ]]; then
    if [[ -s "$delete_list" ]]; then
      while IFS= read -r f; do
        rm -f -- "$f"
      done < "$delete_list"
      echo
      echo "该目录删除完成。"
    else
      echo
      echo "该目录没有需要删除的文件。"
    fi
  fi

  cd "$oldpwd"
  rm -rf "$tmpdir"
}

if [[ "$RECURSIVE" -eq 1 ]]; then
  echo "递归处理根目录：$TARGET_DIR"
  if [[ "$DO_DELETE" -eq 1 ]]; then
    echo "模式：真正删除"
  else
    echo "模式：预览，不删除"
  fi

  mapfile -d '' dirs < <(
    find "$TARGET_DIR" -type f -name '*.pth' -printf '%h\0' \
      | sort -zu
  )

  if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "没有找到 .pth 文件。"
    exit 0
  fi

  for dir in "${dirs[@]}"; do
    clean_one_dir "$dir"
  done
else
  echo "处理目录：$TARGET_DIR"
  if [[ "$DO_DELETE" -eq 1 ]]; then
    echo "模式：真正删除"
  else
    echo "模式：预览，不删除"
  fi

  clean_one_dir "$TARGET_DIR"
fi

echo
if [[ "$DO_DELETE" -eq 1 ]]; then
  echo "全部处理完成。"
else
  echo "当前是预览模式，没有删除任何文件。"
  if [[ "$RECURSIVE" -eq 1 ]]; then
    echo "确认无误后执行：bash clean_pth_recursive.sh \"$TARGET_DIR\" --recursive --delete"
  else
    echo "确认无误后执行：bash clean_pth_recursive.sh \"$TARGET_DIR\" --delete"
  fi
fi
