import json,pathlib,subprocess
root=pathlib.Path('work');out=root/'video-frames';out.mkdir(exist_ok=True)
d=json.loads((root/'video6-metadata-current.json').read_text())
formats=[x for x in d['formats'] if x.get('vcodec','none').startswith('avc') and x.get('height')==720 and x.get('protocol')=='https']
if not formats:raise RuntimeError('No suitable format')
f=formats[0];target=out/'2026-06-3300.png'
r=subprocess.run(['/opt/homebrew/bin/ffmpeg','-loglevel','error','-ss','3300','-i',f['url'],'-frames:v','1','-an','-vf','scale=-2:720','-y',str(target)],capture_output=True,timeout=60)
status={'video_id':d['id'],'timestamp':3300,'source_duration':d['duration'],'format_id':f['format_id'],'success':r.returncode==0 and target.exists(),'returncode':r.returncode}
if r.returncode:status['error_category']='media read or decode failed; raw URL suppressed'
(out/'probe.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
