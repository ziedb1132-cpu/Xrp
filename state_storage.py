"""Split every persisted XRP runtime artifact above 90 MiB and verify on restore."""
import hashlib,json,os,shutil
from pathlib import Path
ARTIFACTS=("xrp_state.json","xrp_summary.json","strategy_history.json","xrp_model.pt","xrp_exit_policy.pt","xrp_live_state.json")
LIMIT=90*1024*1024
BLOCK=1024*1024

def _paths(name):
    source=Path(name)
    return source,Path(name+".parts"),Path(name+".parts.json")

def restore():
    restored=[]
    for name in ARTIFACTS:
        source,parts,manifest=_paths(name)
        if source.exists() or not manifest.exists():continue
        meta=json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("format")!=1 or not isinstance(meta.get("parts"),list):raise RuntimeError("Invalid manifest for "+name)
        tmp=source.with_name(source.name+".restore");h=hashlib.sha256();size=0
        try:
            with tmp.open("wb") as out:
                for part_name in meta["parts"]:
                    part=(parts/part_name).resolve()
                    if parts.resolve() not in part.parents or not part.is_file():raise RuntimeError("Missing or unsafe part: "+part_name)
                    with part.open("rb") as src:
                        for block in iter(lambda:src.read(BLOCK),b""):out.write(block);h.update(block);size+=len(block)
            if size!=meta["size"] or h.hexdigest()!=meta["sha256"]:raise RuntimeError("SHA-256/size verification failed for "+name)
            os.replace(tmp,source);restored.append(name)
        except Exception:
            tmp.unlink(missing_ok=True);raise
    return "restored:"+",".join(restored) if restored else "nothing-to-restore"

def pack():
    results=[]
    for name in ARTIFACTS:
        source,parts,manifest=_paths(name)
        if not source.exists():continue
        if source.stat().st_size<=LIMIT:
            if parts.exists():shutil.rmtree(parts)
            manifest.unlink(missing_ok=True);continue
        if parts.exists():shutil.rmtree(parts)
        parts.mkdir(parents=True);h=hashlib.sha256();names=[];total=0
        with source.open("rb") as src:
            index=0
            while True:
                block=src.read(LIMIT)
                if not block:break
                part_name=f"chunk-{index:04d}.part";(parts/part_name).write_bytes(block)
                names.append(part_name);h.update(block);total+=len(block);index+=1
        data={"format":1,"source":name,"size":total,"sha256":h.hexdigest(),"chunk_size":LIMIT,"parts":names}
        tmp=manifest.with_name(manifest.name+".tmp");tmp.write_text(json.dumps(data,separators=(",",":")),encoding="utf-8")
        os.replace(tmp,manifest);source.unlink();results.append(name+":"+str(len(names))+" parts")
    return "split:"+(",".join(results) if results else "none")

if __name__=="__main__":
    import sys
    action=sys.argv[1] if len(sys.argv)>1 else "restore"
    print(restore() if action=="restore" else pack())
