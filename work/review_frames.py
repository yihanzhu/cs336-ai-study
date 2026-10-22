import concurrent.futures,json,pathlib,subprocess,sys
import yt_dlp
root=pathlib.Path('work');out=root/'video-frames';out.mkdir(exist_ok=True)
jobs=[('2026-01','JuoVZkPBiKk',[4410]),('2026-02','kuYAsz7zspQ',[1510,3840]),('2026-03','lVynu4bo1rY',[2170]),('2026-04','cKSwj_qZ8Jg',[3800]),('2026-05','izZba4UA7iY',[4530]),('2026-06','xnDHaNUvHBg',[2150]),('2026-07','SzpOcwdIL0Y',[2500]),('2026-08','6-cXp-aOmdg',[2270,3590]),('2026-10','EfM546A79aM',[4470]),('2025-14','9Cd0THLS1t0',[3380]),('2025-15','Dfu7vC9jo4w',[3100]),('2025-17','JdGFdViaOJk',[4120])]
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
(out/'frames-final-index.json').write_text(json.dumps(items,indent=2))
