"""Cold source namespace and import audits against the pinned tree and Humming wheel."""
import argparse
import ast
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import symtable
import sys
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent
BASE="297e54fd105e974cdf401e5b4d20253a22c16f90"
TREE="f1ed4dd4f1d567819e193b8c437717424ace0e9d"
parser=argparse.ArgumentParser()
parser.add_argument("stage",type=Path)
parser.add_argument("wheel_dir",type=Path)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
assert sha((args.wheel_dir/"pyflakes.whl").read_bytes())=="f742a7dbd0d9cb9ea41e9a24a918996e8170c799fa528688d40dd582c8265f4f"
vendor=zipfile.ZipFile(args.wheel_dir/"humming-kernels.whl")
assert sha((args.wheel_dir/"humming-kernels.whl").read_bytes())=="cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331"
sys.path.insert(0,str(args.wheel_dir/"pyflakes.whl"))
from pyflakes import checker,messages

def git(*cmd,data=None,env=None):
    return subprocess.check_output(["git","-C",args.stage,*cmd],input=data,env=env)

files=set(git("ls-tree","-r","--name-only",TREE).decode().splitlines())
paths=[p for p in git("diff","--name-only",BASE,TREE).decode().splitlines() if p.endswith(".py")]
oldfiles=set(git("ls-tree","-r","--name-only",BASE).decode().splitlines())

def names(src,p):
    tree=ast.parse(src,filename=p)
    compile(tree,p,"exec")
    # A[...] is the project's Annotated alias. Its Arg(...) string arguments are
    # runtime CLI metadata; Pyflakes heuristically tries parsing those as types.
    arg_strings={(child.lineno,child.col_offset) for node in ast.walk(tree)
        if isinstance(node,ast.Call) and ast.unparse(node.func)=="Arg" for child in ast.walk(node)
        if isinstance(child,ast.Constant) and isinstance(child.value,str)}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and ast.unparse(node.value) == "A"
                and isinstance(node.slice, ast.Tuple)):
            for meta in node.slice.elts[1:]:
                arg_strings.update((x.lineno,x.col_offset) for x in ast.walk(meta)
                                   if isinstance(x,ast.Constant) and isinstance(x.value,str))
    findings=Counter(); cli=0
    for m in checker.Checker(tree,filename=p).messages:
        if not isinstance(m,(messages.UndefinedName,messages.UndefinedLocal,messages.ImportStarUsage,
                             messages.ForwardAnnotationSyntaxError)):continue
        if (m.lineno,m.col) in arg_strings:
            cli+=1
            continue
        findings[(type(m).__name__,str(m.message_args))]+=1
    imports={(node.module,a.name) for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)
             and not node.level and node.module and node.module.startswith(("humming","sglang"))
             for a in node.names}
    return findings,cli,imports


@lru_cache(None)
def get_module(name):
    stem=name.replace(".","/")
    for ending in (".py","/__init__.py"):
        if name.startswith("sglang"):
            p="python/"+stem+ending
            if p in files: return git("show",TREE+":"+p).decode()
        else:
            p=stem+ending
            if p in vendor.namelist():return vendor.read(p).decode()
    return None


def resolves(mod,member):
    raw=get_module(mod)
    if raw is None:return get_module(mod+"."+member) is not None
    syms=symtable.symtable(raw,mod,"exec")
    if member in syms.get_identifiers():
        sym=syms.lookup(member)
        if sym.is_assigned() or sym.is_imported() or sym.is_namespace():return True
    return get_module(mod+"."+member) is not None


results={}; checked=set()
for p in paths:
    old,old_meta,old_imports=names(git("show",BASE+":"+p) if p in oldfiles else b"",p)
    current,current_meta,imports=names(git("show",TREE+":"+p),p)
    novel=current-old
    assert not novel,(p,novel)
    for mod,member in imports-old_imports:
        assert resolves(mod,member),(p,mod,member)
        checked.add((mod,member))
    results[p]=dict(new_name_errors=0,inherited_name_diagnostics=sum((current&old).values()),
        old_literal_metadata_diagnostics=old_meta,new_literal_metadata_diagnostics=current_meta,
        added_project_or_vendor_imports=len(imports-old_imports))
with tempfile.TemporaryDirectory() as tmp:
    env={**os.environ,"GIT_INDEX_FILE":str(Path(tmp)/"index")}
    git("read-tree",BASE,env=env)
    patch=(HERE.parent/"v88_v64_fused_draft_head/patch.diff").read_bytes()
    git("apply","--cached","-",data=patch,env=env)
    p="python/sglang/srt/models/dflash.py"; historical=git("show",":"+p,env=env)
    errors,_,_=names(historical,p)
    assert errors["UndefinedName","('is_cuda',)"]>0

print(json.dumps(dict(base=BASE,tree=TREE,compared_modules=len(paths),added_resolved_imports=len(checked),
    pyflakes="3.4.0",pyflakes_wheel_sha256=sha((args.wheel_dir/"pyflakes.whl").read_bytes()),
    humming="0.1.12",humming_wheel_sha256=sha((args.wheel_dir/"humming-kernels.whl").read_bytes()),
    historical_positive_control="V88 real candidate_topk module: is_cuda was caught",
    results=results),indent=2))
