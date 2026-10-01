set -euo pipefail
export PYTHONUNBUFFERED=1
REPO="/tmp/vcm_autonomous_ar"
CODEC="__CODEC__"
STAGE="__STAGE__"
CONFIG="__CONFIG__"
SEED="__SEED__"
MAX_STEPS="__STEPS__"
SOURCE_SLUG="__SOURCE_SLUG__"
SOURCE_SHA="__SOURCE_SHA__"
SOURCE_FILENAME="__SOURCE_FILENAME__"
ROOT_OUT="/kaggle/working/outputs/autonomous_ar/__SLUG__"
mkdir -p "$ROOT_OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "__SLUG__.tgz" "outputs/autonomous_ar/__SLUG__"
  echo "[exit] rc=$rc job=__SLUG__"
  exit "$rc"
}
trap finish EXIT
git clone -q https://github.com/munnn01/pre_updated_v2.git "$REPO"
git -C "$REPO" checkout -q d39927e92298c111d498f0e1a9f72069469ed0b9
cd "$REPO"
python - "$ROOT_OUT" <<'BOOTSTRAP'
import base64,hashlib,io,json,sys,zipfile
from pathlib import Path
payload=base64.b64decode("__PAYLOAD_B64__")
assert hashlib.sha256(payload).hexdigest()=="__PAYLOAD_SHA__"
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        target=Path(name)
        assert not target.is_absolute() and '..' not in target.parts
        assert target.parts[0] in ('src','configs','ops','docs','tests')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.read(name))
spec=json.loads(base64.b64decode("__SPEC_B64__"))
spec['payload_sha256']=hashlib.sha256(payload).hexdigest()
Path(sys.argv[1],'job_spec.json').write_text(json.dumps(spec,indent=2,sort_keys=True))
print('[payload]',spec['candidate_ref'],spec['payload_sha256'])
BOOTSTRAP
python -c 'import torch,torchvision; assert torch.cuda.is_available(); print("torch",torch.__version__,"torchvision",torchvision.__version__,"gpu",torch.cuda.get_device_name(0))'
KIN_ROOT=""
for candidate in /kaggle/input/kineticscleaned /kaggle/input/datasets/qktttttttttt/kineticscleaned; do
  if [ -d "$candidate" ]; then KIN_ROOT="$candidate"; break; fi
done
if [ -z "$KIN_ROOT" ]; then echo 'ERROR: locked Kinetics dataset missing' >&2; exit 2; fi
python scripts/build_train_index.py --root "$KIN_ROOT" --out "$ROOT_OUT/full_index.json" --assert-fingerprint 30f083f8520a
python ops/autonomous_ar/protocol.py indices --index "$ROOT_OUT/full_index.json" --out "$ROOT_OUT/indices"
SOURCE="$(find /kaggle/input -type f -path "*${SOURCE_SLUG}*" -name "$SOURCE_FILENAME" -print -quit || true)"
if [ -z "$SOURCE" ]; then echo 'ERROR: audited source checkpoint missing' >&2; exit 3; fi
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA"
python - "$SOURCE" "$STAGE" "$MAX_STEPS" <<'SOURCE_AUDIT'
import sys,torch
path,stage,steps=sys.argv[1:]
state=torch.load(path,map_location='cpu',weights_only=False)
assert state['global_step']==(500 if stage=='screen' else int(steps))
assert state['cfg']['model']['arch']=='additive_cond'
assert state['cfg']['model']['cond_dim']==3
print('[source]',stage,state['global_step'])
SOURCE_AUDIT
if [ "$STAGE" = screen ]; then
  mkdir -p "$ROOT_OUT/train/checkpoints"
  cp "$SOURCE" "$ROOT_OUT/train/checkpoints/preprocessor.pth"
  python train.py --config "$CONFIG" data.index="$ROOT_OUT/indices/dev.json" out_dir="$ROOT_OUT/train" seed="$SEED" \
    codec.ste_codec="$CODEC" codec.ste_alternate=false train.epochs=1 train.max_steps="$MAX_STEPS" \
    train.finetune=true train.resume=false train.cosine=false train.patience=0 \
    2>&1 | tee "$ROOT_OUT/train/train.log"
  FINAL="$ROOT_OUT/train/checkpoints/preprocessor_last.pth"
  PHASES="dev"
else
  FINAL="$SOURCE"
  PHASES="confirmation full"
fi
python - "$FINAL" "$ROOT_OUT" "$CODEC" "$MAX_STEPS" <<'FINAL_AUDIT'
import hashlib,json,math,sys,torch
from pathlib import Path
path,out,codec,steps=sys.argv[1:]
state=torch.load(path,map_location='cpu',weights_only=False)
assert state['global_step']==int(steps)
assert state['cfg']['codec']['ste_codec']==codec
assert state['cfg']['codec']['ste_alternate'] is False
assert state['cfg']['loss']['task_regret_constraint']['enabled'] is True
assert state['cfg']['loss']['rate_constraint']['enabled'] is False
assert state['cfg']['train']['finetune'] is True and state['cfg']['train']['resume'] is False
assert set(state['task_duals'])=={f'{codec}:{q}' for q in (30,35,40,45,50)}
assert all(math.isfinite(float(v)) and 0<=float(v)<=state['cfg']['loss']['task_regret_constraint']['mu_max'] for v in state['task_duals'].values())
manifest={'final_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'global_step':state['global_step'],'epoch':state['epoch'],'task_duals':state['task_duals'],'cfg':state['cfg']}
Path(out,'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
print('[final]',manifest['global_step'],manifest['final_sha256'])
FINAL_AUDIT
for PHASE in $PHASES; do
  mkdir -p "$ROOT_OUT/$PHASE"
  python evaluate.py --config "$CONFIG" --ckpt "$FINAL" --out "$ROOT_OUT/$PHASE" \
    data.index="$ROOT_OUT/indices/$PHASE.json" eval.split=val eval.codecs="[$CODEC]" \
    eval.shard_idx=null eval.num_shards=1 eval.shard_salt=null eval.per_sequence=true \
    eval.include_proxy=false eval.held_out_backbone=r2plus1d_18 \
    2>&1 | tee "$ROOT_OUT/$PHASE/eval.log"
  python ops/autonomous_ar/protocol.py score --index "$ROOT_OUT/full_index.json" --codec "$CODEC" --stage "$PHASE" \
    --results "$ROOT_OUT/$PHASE/results.json" --points "$ROOT_OUT/$PHASE/sequence_points.csv" --out "$ROOT_OUT/$PHASE/decision.json"
done
python - "$ROOT_OUT" "$STAGE" <<'DECISION'
import json,sys
from pathlib import Path
out,stage=Path(sys.argv[1]),sys.argv[2]
phases=['dev'] if stage=='screen' else ['confirmation','full']
scores={phase:json.loads((out/phase/'decision.json').read_text()) for phase in phases}
summary={'stage':stage,'scores':scores,'development_pass':scores.get('dev',{}).get('passes',False),'goal_complete':stage=='confirmation' and all(row['passes'] for row in scores.values())}
(out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
print('[result]',json.dumps(summary,sort_keys=True))
DECISION
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA"
echo "[done] __SLUG__"
