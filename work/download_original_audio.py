import hashlib,json,pathlib
import yt_dlp
root=pathlib.Path('work/missing-audio');root.mkdir(exist_ok=True)
class Quiet:
 def debug(self,*a):pass
 def warning(self,*a):pass
 def error(self,*a):pass
record={'video_id':'6-cXp-aOmdg','success':False}
try:
 opts={'quiet':True,'no_warnings':True,'logger':Quiet(),'skip_download':True,'socket_timeout':15,'retries':0,'extractor_retries':0}
 with yt_dlp.YoutubeDL(opts) as y:d=y.extract_info('https://www.youtube.com/watch?v=6-cXp-aOmdg',download=False)
 candidates=[x for x in d['formats'] if x.get('vcodec')=='none' and x.get('language','').startswith('en') and x.get('ext')=='m4a' and x.get('protocol')=='https']
 candidates.sort(key=lambda x:x.get('abr') or 999);f=candidates[0]
 assert 'original' in f.get('format_note','').lower()
 record.update(duration=d['duration'],language=f['language'],format_id=f['format_id'],format_note=f['format_note'])
 print('Original English audio selected:',f['format_id'],f['language'],flush=True)
 opts.update(skip_download=False,format=f['format_id'],http_chunk_size=1048576,outtmpl=str(root/'2026-08-original.%(ext)s'),noprogress=True,overwrites=True)
 with yt_dlp.YoutubeDL(opts) as y:y.download(['https://www.youtube.com/watch?v=6-cXp-aOmdg'])
 p=root/'2026-08-original.m4a';record.update(success=p.exists(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
except Exception as e:record['error_type']=type(e).__name__
(root/'source-original.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
