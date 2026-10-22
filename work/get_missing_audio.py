import json,pathlib,subprocess,hashlib,re
import yt_dlp
root=pathlib.Path('work');out=root/'missing-audio';out.mkdir(exist_ok=True)
class Quiet:
 def debug(self,*a):pass
 def warning(self,*a):pass
 def error(self,*a):pass
with yt_dlp.YoutubeDL({'quiet':True,'no_warnings':True,'logger':Quiet(),'skip_download':True,'socket_timeout':20,'retries':0,'extractor_retries':0}) as y:
 d=y.extract_info('https://www.youtube.com/watch?v=6-cXp-aOmdg',download=False)
formats=[x for x in d['formats'] if x.get('vcodec')=='none' and x.get('acodec')!='none' and x.get('protocol')=='https']
formats.sort(key=lambda x:(x['format_id']!='140',not x.get('acodec','').startswith('mp4a'),abs((x.get('abr') or 80)-64)))
f=formats[0];p=out/'2026-08.wav'
r=subprocess.run(['/opt/homebrew/bin/ffmpeg','-loglevel','error','-i',f['url'],'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le','-y',str(p)],capture_output=True,timeout=240)
record={'video_id':d['id'],'duration':d['duration'],'success':r.returncode==0 and p.exists(),'format_id':f['format_id']}
if record['success']:record.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
else:record['error']=re.sub(r'https?://\S+','<media-url>',r.stderr.decode(errors='replace'))[-1200:]
(out/'source.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
