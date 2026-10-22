import json,pathlib
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
out=pathlib.Path('work/reacquisition');out.mkdir(exist_ok=True)
class Quiet:
 def debug(self,*a):pass
 def warning(self,*a):pass
 def error(self,*a):pass
for video in ['r3DC_gjFCSA','6-cXp-aOmdg']:
 r={'video_id':video,'checked_date':'2026-10-22'}
 try:
  with yt_dlp.YoutubeDL({'quiet':True,'no_warnings':True,'logger':Quiet(),'skip_download':True,'socket_timeout':15,'retries':0,'extractor_retries':0}) as y:d=y.extract_info('https://www.youtube.com/watch?v='+video,download=False)
  r['metadata']={k:d.get(k) for k in ['title','description','channel','channel_id','uploader','upload_date','duration','webpage_url','availability']}
 except Exception as e:r['metadata_error']=type(e).__name__
 try:
  tracks=YouTubeTranscriptApi().list(video)
  r['tracks']=[{'language_code':t.language_code,'is_generated':t.is_generated} for t in tracks]
  t=tracks.find_transcript(['en-US','en','en-GB']);data=t.fetch()
  r['selected_language']=data.language_code;r['is_generated']=data.is_generated;r['segments']=data.to_raw_data()
 except Exception as e:r['caption_error']=type(e).__name__
 (out/(video+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in r.items() if k not in ['segments','tracks']},ensure_ascii=False),flush=True)
