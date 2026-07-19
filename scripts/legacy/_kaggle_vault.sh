#!/usr/bin/env bash
set -uo pipefail
BASE="/c/Users/sharg/Desktop/next mission/RADAR AudioShield/datasets"
STAGE_ROOT="$BASE/_vault_staging"; LOG="$BASE/_kaggle_vault.log"
VAULT=(05_AI4T 02_In-the-Wild 07_FakeOrReal 04_ReplayDF)   # smallest first
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
KUSER="${KAGGLE_USERNAME:?set KAGGLE_USERNAME=your_kaggle_username and re-run}"; echo "kaggle user: $KUSER"
vault_one(){
  local name="$1" src="$BASE/$1"
  local slug="radar-vault-$(echo "$name" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')"
  local stage="$STAGE_ROOT/$slug" marker="$src/_VAULTED_OK"
  [ -f "$marker" ] && { log "[$name] already vaulted — skip"; return 0; }
  [ -s "$src/_SHA256.txt" ] || { log "[$name] REFUSED: no _SHA256.txt"; return 1; }
  local need_kb free_kb; need_kb=$(du -sk "$src" | cut -f1)
  free_kb=$(df -P "$BASE" | awk 'NR==2{print $4}')
  [ "$free_kb" -gt $((need_kb + 5242880)) ] || { log "[$name] REFUSED: not enough free disk for staging"; return 1; }
  log "[$name] staging (chunked tar)..."
  rm -rf "$stage"; mkdir -p "$stage"
  ( cd "$src" && tar -cf - --exclude='_*' . ) | ( cd "$stage" && split -b 9500m -d - "${name}.tar.part-" ) || { rm -rf "$stage"; return 1; }
  cp "$src/_SHA256.txt" "$stage/"; ( cd "$stage" && sha256sum "${name}".tar.part-* > _CHUNKS_SHA256.txt )
  printf '{"title":"RADAR vault %s","id":"%s/%s","licenses":[{"name":"other"}]}\n' "$name" "$KUSER" "$slug" > "$stage/dataset-metadata.json"
  log "[$name] uploading -> kaggle.com/$KUSER/$slug (PRIVATE)..."
  if kaggle datasets create -p "$stage" >>"$LOG" 2>&1 || kaggle datasets version -p "$stage" -m "refresh $(date '+%F')" >>"$LOG" 2>&1; then
    sleep 20
    local remote; remote=$(kaggle datasets files "$KUSER/$slug" 2>>"$LOG" | tail -n +2 | grep -c .)
    local staged; staged=$(ls "$stage" | grep -vc 'dataset-metadata')
    log "[$name] remote=$remote staged=$staged"
    if [ "$remote" -ge "$staged" ]; then
      echo "$KUSER/$slug @ $(date '+%F %T')" > "$marker"; rm -rf "$stage"; log "[$name] VAULTED OK"; return 0
    fi
    log "[$name] listing mismatch — staging kept; retry: bash _kaggle_vault.sh $name"; return 1
  fi
  log "[$name] upload FAILED — staging kept; retry: bash _kaggle_vault.sh $name"; return 1
}
mkdir -p "$STAGE_ROOT"; touch "$LOG"; FAIL=()
if [ $# -ge 1 ]; then T=("$@"); else T=("${VAULT[@]}"); fi
for c in "${T[@]}"; do vault_one "$c" || FAIL+=("$c"); done
log "=== done ==="; [ ${#FAIL[@]} -gt 0 ] && { log "RETRY NEEDED: ${FAIL[*]}"; exit 1; }
log "All vaulted. Eval-trio deletion gate 2 satisfied. ASVspoof5/DiffSSD/VCTK still await step-3 embeddings gate."
