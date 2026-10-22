import json,pathlib,traceback
from youtube_transcript_api import YouTubeTranscriptApi
out={}
video='JuoVZkPBiKk'
try:
    api=YouTubeTranscriptApi()
    tracks=api.list(video)
    out['tracks']=[{'language':t.language_code,'generated':t.is_generated} for t in tracks]
    selected=tracks.find_transcript(['en-US','en','en-GB'])
    snippets=selected.fetch()
    out['transcript']={'video_id':video,'language':snippets.language_code,'generated':snippets.is_generated,'segments':snippets.to_raw_data()}
except Exception as e:
    out['error']={'type':type(e).__name__,'message':str(e)}
pathlib.Path('work/transcript-probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
if 'transcript' in out: print('SUCCESS',len(out['transcript']['segments']),'segments')
else: print(json.dumps(out,ensure_ascii=False,indent=2))
