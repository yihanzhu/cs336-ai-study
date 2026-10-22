import concurrent.futures,json,pathlib,subprocess,sys
import yt_dlp
root=pathlib.Path('work');out=root/'video-frames';out.mkdir(exist_ok=True)
jobs=[('2026-06','xnDHaNUvHBg',[1700,3080]),('2026-07','SzpOcwdIL0Y',[2470]),('2026-14','5sxHosTLPF8',[2370]),('2026-18','9EEm4iMAF5s',[2370,3020]),('2025-17','JdGFdViaOJk',[2140,3070])]
if len(sys.argv)>1 and sys.argv[1]=='missing':jobs=[('2026-08','6-cXp-aOmdg',[1120,2660])]
class Quiet:
 def debug(self,*a):pass
 def warning(self,*a):pass
 def error(self,*a):pass
def run(job):
 label,video,times=job;results=[]
 try:
  if video=='xnDHaNUvHBg':d=json.loads((root/'video6-metadata-current.json').read_text())
  else:
   with yt_dlp.YoutubeDL({'quiet':True,'no_warnings':True,'logger':Quiet(),'skip_download':True,'socket_timeout':15,'retries':0,'extractor_retries':0}) as y:d=y.extract_info('https://www.youtube.com/watch?v='+video,download=False)
  f=next(x for x in d['formats'] if x.get('vcodec','none').startswith('avc') and x.get('height')==720 and x.get('protocol')=='https')
  for t in times:
   p=out/f'{label}-{t}.png'
   r=subprocess.run(['/opt/homebrew/bin/ffmpeg','-loglevel','error','-ss',str(t),'-i',f['url'],'-frames:v','1','-an','-vf','scale=-2:720','-y',str(p)],capture_output=True,timeout=60)
   results.append({'lecture':label,'video_id':video,'timestamp':t,'success':r.returncode==0 and p.exists(),'path':str(p),'source_duration':d['duration']})
 except Exception as e:results.append({'lecture':label,'video_id':video,'success':False,'error_type':type(e).__name__})
 return results
items=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for r in pool.map(run,jobs):items+=r;print(json.dumps(r),flush=True)
(out/('frames8-index.json' if len(sys.argv)>1 else 'frames-index.json')).write_text(json.dumps(items,indent=2))
