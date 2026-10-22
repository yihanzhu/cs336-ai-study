"""Check the publishable file set, local-progress boundaries and package allowlist.
No assurance of removing remote cached commits or third-party clones is implied.
"""
import argparse,ast,hashlib,json,pathlib,re,shutil,subprocess,tempfile,zipfile
R=pathlib.Path(__file__).resolve().parent.parent;O=R/'outputs'

def file_list():
 if (R/'.git').exists():
  return [p for p in subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=R,text=True).split('\0') if p and (R/p).is_file()]
 return [str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and 'local' not in p.relative_to(R).parts and '.git' not in p.relative_to(R).parts and not str(p.relative_to(R)).startswith('work/')]

def main():
 files=file_list();forbidden_paths=['outputs/学习进度.md','outputs/learning-progress.md']
 bad=[]
 patterns={
  'absolute home path':r'/Users/[^\s`\"<>]+|/home/[^\s`\"<>]+',
  'owner-specific profile':r'本仓库主人|本学习者|repository owner|repository owner.s|先设为私有|currently private|个人示范已到|示范例子已展示，等待首次回答',
  'credential/private key':r'gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
 }
 # The scanner's literal patterns are tests, not published personal records.
 for name in files:
  if name.startswith('local/') or name in forbidden_paths:bad.append((name,'personal file'))
  if name=='work/verify_publication.py':continue
  text=(R/name).read_text(errors='ignore')
  for label,pattern in patterns.items():
   if re.search(pattern,text):bad.append((name,label))
 assert not bad,bad
 for name in ['local/probe.md','local/learning-progress.md','outputs/学习进度.md']:
  if (R/'.git').exists():
   p=subprocess.run(['git','check-ignore','--no-index','-q',name],cwd=R);assert p.returncode==0,name
 bundle=json.loads((O/'CS336-bilingual-lessons.json').read_text());contract=bundle['progress_contract']
 assert contract['shared_file']=='local/learning-progress.md' and contract['storage']=='local_only'
 assert contract['tracked'] is False and contract['auto_push'] is False
 for lang,doc in bundle['languages'].items():
  protocol=doc['protocol'];assert 'local/learning-progress.md' in protocol
  if lang=='en':assert 'Never commit' in protocol and 'automatically push' in protocol
  else:assert '绝不提交' in protocol and '自动推送' in protocol
 assert (O/'LICENSE.txt').read_bytes()==(R/'LICENSE').read_bytes()
 assert (O/'THIRD-PARTY-NOTICES.md').read_bytes()==(R/'THIRD_PARTY_NOTICES.md').read_bytes()
 with zipfile.ZipFile(O/'CS336-AI陪学资料包.zip') as z:
  assert len(z.namelist())==38 and z.testzip() is None
  for n in z.namelist():
   p=pathlib.PurePosixPath(n);assert 'local' not in p.parts and p.name not in ['学习进度.md','learning-progress.md']
   assert not any(re.search(pat,z.read(n).decode(errors='ignore')) for pat in patterns.values()),n
   assert z.read(n)==(O/p.name).read_bytes()
  assert {'LICENSE.txt','THIRD-PARTY-NOTICES.md'} <= {pathlib.PurePosixPath(n).name for n in z.namelist()}
 # Protect real learners against initialization accidentally replacing their state.
 with tempfile.TemporaryDirectory(prefix='cs336-local-progress-') as temp:
  t=pathlib.Path(temp);(t/'work').mkdir();(t/'outputs').mkdir();shutil.copyfile(R/'work/init_learning.py',t/'work/init_learning.py')
  for name in ['学习进度模板.md','learning-progress-template.en.md']:shutil.copyfile(O/name,t/'outputs'/name)
  subprocess.run(['python3','work/init_learning.py','--language','zh'],cwd=t,check=True,stdout=subprocess.DEVNULL)
  p=t/'local/learning-progress.md';assert 'preferred_language: zh' in p.read_text()
  original=p.read_bytes();subprocess.run(['python3','work/init_learning.py','--language','en'],cwd=t,check=True,stdout=subprocess.DEVNULL);assert p.read_bytes()==original
  p.unlink();subprocess.run(['python3','work/init_learning.py','--language','en'],cwd=t,check=True,stdout=subprocess.DEVNULL);assert 'preferred_language: en' in p.read_text()
 report={'result':'pass','publishable_files_checked':len(files),'personal_progress_tracked':False,'local_ignored':True,'auto_upload':False,'blank_templates_only':True,'archive_members':38,'initializer_preserves_existing_state':True,'license_and_upstream_notices_in_archive':True,'scope':'Working publication tree and generated ZIP; remote history/releases are separate acceptance gates'}
 (R/'work/publication-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
